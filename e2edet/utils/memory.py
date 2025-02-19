# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from contextlib import contextmanager
from functools import wraps
import torch

from .params import recursive_copy_to_device
from .distributed import synchronize

__all__ = ["retry_if_cuda_oom"]


@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.
    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.
    Args:
        func: a stateless callable that takes tensor-like objects as arguments
    Returns:
        a callable which retries `func` if OOM is encountered.
    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU
    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.
        2. Since the function might be called more than once, it has to be
           stateless.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        print(
            "Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func))
        )
        new_args = (
            recursive_copy_to_device(x, non_blocking=True, device=torch.cpu())
            for x in args
        )
        new_kwargs = {
            k: recursive_copy_to_device(v, non_blocking=True, device=torch.cpu())
            for k, v in kwargs.items()
        }
        synchronize()

        return func(*new_args, **new_kwargs)

    return wrapped
