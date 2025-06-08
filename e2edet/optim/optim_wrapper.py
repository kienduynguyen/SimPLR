# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Generic, Iterator, TypeVar

import torch
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from e2edet.module.parallelize.checkpoint import (
    set_optimizer_state_dict,
    get_optimizer_state_dict,
    StateDictOptions,
)

__all__ = [
    "OptimizersContainer",
    "build_optimizers",
]


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Stateful, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    optimizers: list[T]

    def __init__(
        self,
        model,
        param_groups,
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        self.optimizers = []
        self.model = model

        self.optimizers.append(optimizer_cls(param_groups, **optimizer_kwargs))

        self._validate_length(1)
        self._post_init(param_groups, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        sd = get_optimizer_state_dict(
            self.model,
            self.optimizers,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for k, v in sd.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_optimizer_state_dict(
            self.model,
            self.optimizers,
            state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(self, param_groups, optimizer_kwargs: dict[str, Any]) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, param_groups, optimizer_kwargs)


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    def __init__(
        self,
        model,
        param_groups,
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        self.model = model

        flatten_param_groups = []
        for param_group in param_groups:
            for param in param_group["params"]:
                new_param_group = {}
                for kk, vv in param_group.items():
                    if kk != "params":
                        new_param_group[kk] = vv
                new_param_group["params"] = [param]
                flatten_param_groups.append(new_param_group)

        optim_dict = {}
        for param_group in flatten_param_groups:
            assert len(param_group["params"]) == 1
            optim_dict[param_group["params"][0]] = optimizer_cls(
                [param_group], **optimizer_kwargs
            )

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for param_group in flatten_param_groups:
            param_group["params"][0].register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())

        self._validate_length(len(list(model.parameters())))
        self._post_init(flatten_param_groups, optimizer_kwargs)

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass
