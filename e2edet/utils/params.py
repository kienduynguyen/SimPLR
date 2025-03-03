from collections import abc
from math import inf
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist


def get_global_rank(group: Any, rank: int) -> int:
    if group is dist.group.WORLD:
        return rank

    return dist.distributed_c10d._get_global_rank(group, rank)


def recursive_copy_to_device(
    value: Any,
    non_blocking: bool,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Any:
    """
    Recursively searches lists, tuples, dicts and copies tensors to device if
    possible. Non-tensor values are passed as-is in the result.
    NOTE:  These are all copies, so if there are two objects that reference
    the same object, then after this call, there will be two different objects
    referenced on the device.
    """

    if isinstance(value, torch.Tensor):
        if value.dtype == torch.float16 or value.dtype == torch.bfloat16:
            return value.to(device, dtype=dtype, non_blocking=non_blocking)
        else:
            return value.to(device, non_blocking=non_blocking)

    if isinstance(value, (list, tuple)):
        values = []
        for val in value:
            values.append(
                recursive_copy_to_device(
                    val, non_blocking=non_blocking, device=device, dtype=dtype
                )
            )

        return values if isinstance(value, list) else tuple(values)

    if isinstance(value, abc.Mapping):
        device_val: Dict[str, Any] = {}
        for key, val in value.items():
            device_val[key] = recursive_copy_to_device(
                val, non_blocking=non_blocking, device=device, dtype=dtype
            )

        return device_val

    return value


def calc_grad_norm(parameters: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    r"""Calculate gradient norm of an iterable of parameters.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda par: par.grad is not None, parameters))

    if len(parameters) == 0:
        return torch.tensor(0.0)
    p = float(p)
    if p == inf:
        local_norm = max(par.grad.detach().abs().max() for par in parameters)  # type: ignore
    else:
        # Compute the norm in full precision no matter what
        local_norm = torch.norm(torch.stack([torch.norm(par.grad.detach(), p, dtype=torch.float32) for par in parameters]), p).to(dtype=parameters[0].dtype)  # type: ignore
    return local_norm


class Workhandle:
    def __init__(self, handle: Any, callback: Optional[Callable]) -> None:
        self.handle = handle
        self.callback = callback


class Bucket:
    """
    Helper class to simplify the handling of buckets, which unify the underlying storage of multiple tensors
    """

    def __init__(self, size: int, dtype: torch.dtype, device: torch.device) -> None:
        self._params: List[torch.Tensor] = []
        self._param_ids: List[int] = []
        self._fill = 0

        # The actual flat tensor
        self.buffer: torch.Tensor = torch.zeros(size, dtype=dtype, device=device)

    def to(  # type: ignore
        self,
        device: Optional[Union[int, torch.device]],
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        keep_param_alignment: bool = True,
    ) -> "ParamBucket":
        """
        Move the underlying buffer
        """
        assert (
            self.buffer is not None
        ), "Cannot move a collapsed bucket, please rebuild it"
        self.buffer = self.buffer.to(device, dtype, non_blocking)


class ParamBucket(Bucket):
    """
    Helper class to simplify the handling of parameter buckets
    """

    def __init__(self, size: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(size, dtype, device)

    def to(  # type: ignore
        self,
        device: Optional[Union[int, torch.device]],
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        keep_param_alignment: bool = True,
    ) -> "ParamBucket":
        """
        Move the underlying buffer
        """
        super().to(device, dtype, non_blocking)

        if keep_param_alignment:
            self._reattach_params()

    @torch.no_grad()
    def add_param(self, param: torch.Tensor) -> None:
        """
        Add a new parameter gradient to the bucket. Param becomes a view of this bucket buffer
        """

        assert (
            id(param) not in self._param_ids
        ), "The same param cannot be checked in twice"

        self._add_param_as_view(param)
        self._params.append(param)
        self._param_ids.append(id(param))

    @torch.no_grad()
    def _add_param_as_view(
        self, param: torch.Tensor, keep_existing_value: bool = True
    ) -> None:
        assert self.buffer is not None
        assert (
            param.dtype == self.buffer.dtype
        ), f"Different types for the bucket and the param, cannot proceed: {param.dtype} - {self.buffer.dtype}"
        assert (
            param.device == self.buffer.device
        ), f"Different devices for the bucket and the param, cannot proceed: {param.device} - {self.buffer.device}"

        fill_next = self._fill + param.numel()
        assert fill_next <= self.buffer.numel()

        # Copy the current param value
        if keep_existing_value:
            self.buffer[self._fill : fill_next].copy_(param.data.flatten())
        param.data = self.buffer[self._fill : fill_next].view_as(param.data)
        self._fill = fill_next

    @torch.no_grad()
    def _reattach_params(self) -> None:
        """
        Given the parameters which have been registered previously, rebuild the whole bucket
        """
        assert len(self._params) > 0

        self._fill = 0
        for p in self._params:
            if p.dtype != self.buffer.dtype:
                p.data = p.data.to(self.buffer.dtype)
            self._add_param_as_view(p, keep_existing_value=False)


class GradBucket(Bucket):
    """
    Helper class to simplify the handling of gradient buckets
    """

    def __init__(
        self, size: int, dtype: torch.dtype, device: torch.device, destination: int
    ) -> None:
        super().__init__(size, dtype, device)

        self._max_size = size
        self._is_collapsed = False

        self.params_checked_in = 0
        self.destination = destination
        self.sent = True
        self.callback: Optional[Callable[[Any], None]] = None

    def reset_checked_in(self) -> None:
        """Reset the counter of the parameter grads which have been checked in"""
        self.params_checked_in = 0
        self.sent = False

    @property
    def all_checked_in(self) -> bool:
        """Have all the expected gradient check-in happened ?"""
        return len(self._params) == self.params_checked_in

    def can_add_grad_view(self, param: torch.Tensor) -> bool:
        """Is there enough room in the bucket to add this parameter gradient, and is this param not already checked in ?"""
        return (
            self._fill + param.numel() < self._max_size
            and id(param) not in self._param_ids
        )

    def to(  # type: ignore
        self,
        device: Optional[Union[int, torch.device]],
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
        keep_param_alignment: bool = True,
    ) -> "GradBucket":
        """
        Move the underlying buffer
        """
        if self._is_collapsed:
            self.rebuild()

        super().to(device, dtype, non_blocking)

        if keep_param_alignment:
            self._reattach_grads()

    def zero(self) -> None:
        """
        Set all the grads to zero
        """
        self.buffer.fill_(0.0)

    @torch.no_grad()
    def add_grad(self, param: torch.Tensor) -> None:
        """
        Add a new parameter gradient to the bucket. Param.grad becomes a view of this bucket buffer
        """

        assert (
            id(param) not in self._param_ids
        ), "The same gradients cannot be checked in twice"

        if param.grad is None:
            param.grad = torch.zeros_like(param)

        self._add_grad_as_view(param)
        self._params.append(param)
        self._param_ids.append(id(param))

    @torch.no_grad()
    def collapse(self) -> None:
        """
        Release the buffer from memory. The bucket will need to be rebuilt before use
        """
        if not self._is_collapsed:
            for p in self._params:
                assert p.grad is not None
                p.grad.detach_()
                p.grad = None

            self.buffer = torch.zeros(
                0, dtype=self.buffer.dtype, device=self.buffer.device
            )
            self._fill = 0
            self.params_checked_in = 0
            self._is_collapsed = True

    @torch.no_grad()
    def rebuild(self) -> None:
        """
        Given the parameter gradients which have been registered previously, rebuild the whole bucket
        """
        assert len(self._params) > 0

        if self._is_collapsed:
            self.buffer = torch.zeros(
                self._max_size,
                dtype=self._params[0].dtype,
                device=self._params[0].device,
            )

            for p in self._params:
                self._add_grad_as_view(p)

            self._is_collapsed = False

    @torch.no_grad()
    def shrink(self) -> None:
        """
        Shrink the buffer to the size of the parameter gradients currently checked in, release the extra memory
        """
        assert (
            self.buffer.numel() > 0
        ), "Cannot shrink a collapsed bucket, please rebuild"

        self.buffer = self.buffer.resize_(self._fill).clone()
        self._fill = 0
        for p in self._params:
            self._add_grad_as_view(p)

        self._max_size = self._fill

    @torch.no_grad()
    def _reattach_grads(self) -> None:
        """
        Given the parameters gradients which have been registered previously, rebuild the whole bucket
        """
        assert len(self._params) > 0

        self._fill = 0
        for p in self._params:
            self._add_grad_as_view(p, keep_existing_value=False)

    @torch.no_grad()
    def _add_grad_as_view(
        self, param: torch.Tensor, keep_existing_value: bool = True
    ) -> None:
        assert (
            self.buffer.numel() > 0
        ), "Cannot add a gradient to a collapsed bucket, please rebuild"
        assert param.dtype == self.buffer.dtype
        assert param.device == self.buffer.device

        fill_next = self._fill + param.numel()
        assert fill_next <= self.buffer.numel()

        # Copy the current grad value, if any
        if param.grad is not None:
            # keep param.grad in place
            if keep_existing_value:
                self.buffer[self._fill : fill_next].copy_(param.grad.data.flatten())
            param.grad.data = self.buffer[self._fill : fill_next].view_as(param.data)
        else:
            param.grad = self.buffer[self._fill : fill_next].view_as(param.data)
        self._fill = fill_next
