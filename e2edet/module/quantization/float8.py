# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from functools import partial
from typing import Literal, List, Protocol, Union, Optional
from dataclasses import field

import torch
import torch.nn as nn

from nn.parallelize import ParallelDims
from util.misc import has_cuda_capability, module_filter_fn


class ModelConverter(Protocol):
    """General model converter interface.

    A model converter is applying a modification to PyTorch model.
    Typical use cases are:
        - Quantization: using QAT, FP8, ... specialized linear layers;
        - Fused optimized layers (e.g. flash-attention, norms, ...)
    """

    def __init__(self, *, parallel_dims: ParallelDims): ...

    def convert(self, model: nn.Module):
        """Inplace convertion of the model."""
        ...

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        """Post-optimizer (optional) hook (e.g. compute weights statistics)."""
        ...


class Float8Converter(ModelConverter):
    def __init__(
        self,
        enable_fsdp_float8_all_gather: bool,
        precompute_float8_dynamic_scale_for_fsdp: bool,
        force_recompute_fp8_weight_in_bwd: bool,
        parallel_dims: ParallelDims,
        recipe_name: Optional[
            Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"]
        ] = None,
        filter_fqns: list[str] = field(default_factory=list),
    ):
        """
        enable_fsdp_float8_all_gather: bool
            Whether enable float8 all-gather in FSDP, recommended for tensorwise scaling
        precompute_float8_dynamic_scale_for_fsdp: bool
            Whether precompute float8 scales dynamically for FSDP, recommended for tensorwise scaling
        force_recompute_fp8_weight_in_bwd: bool
            Whether to force the recomputation of FP8 weights during backward pass.
            When using FSDP with tensorwise scaling, it is recommended to enable
            `force_recompute_fp8_weight_in_bwd` to prevent saving unsharded FP8 weights
            for backward computation.
        recipe_name: List[str]
            If specified, creates float8 config from recipe name
        filter_fqns: List[str]
            Comma-separated list of fully qualified names of modules to skip applying float8 training to.
            nn.Linear modules with any dim size not divisible by 16 are always skipped due to hardware requirements.
            Example: --float8.filter_fqns "attention.wq,attention.wk,attention.wv,output"
        """
        self.enabled = False

        if not has_cuda_capability(8, 9):
            warnings.warn(
                "Failed to swap to Float8Linear because float8 is only supported on SM89 or later",
            )
            return
        try:
            from torchao.float8 import Float8LinearConfig
        except ImportError as e:
            raise ImportError(
                "torchao is not installed. Please install it to use float8 linear layers."
            ) from e

        if recipe_name is not None and not hasattr(
            Float8LinearConfig, "from_recipe_name"
        ):
            warnings.warn(
                "Failed to swap to Float8Linear with recipe lookup because the torchao version "
                "is too old, please install torchao v0.9.0 or later and try again",
            )
            return

        self.enabled = True
        self.filter_fqns = filter_fqns

        if recipe_name is not None:
            assert (
                not enable_fsdp_float8_all_gather
            ), "using `enable_fsdp_float8_all_gather` together with `recipe_name` is not supported"
            assert (
                not force_recompute_fp8_weight_in_bwd
            ), "using `force_recompute_fp8_weight_in_bwd` together with `recipe_name` is not supported"
            self.config = Float8LinearConfig.from_recipe_name(recipe_name)
            self.precompute_scale = False
            print(f"Float8 training active with recipe {recipe_name}")

            # short-term solution for https://github.com/pytorch/pytorch/issues/150859
            if recipe_name == "rowwise":
                torch._inductor.config.emulate_precision_casts = True
                print("Set torch._inductor.config.emulate_precision_casts to True")

        else:
            # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
            enable_fsdp_float8_all_gather = (
                parallel_dims.dp_shard_enabled and enable_fsdp_float8_all_gather
            )
            self.config = Float8LinearConfig(
                enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
                force_recompute_fp8_weight_in_bwd=force_recompute_fp8_weight_in_bwd,
            )
            # for precompute_float8_dynamic_scale_for_fsdp
            self.precompute_scale = (
                enable_fsdp_float8_all_gather
                and precompute_float8_dynamic_scale_for_fsdp
            )
            print("Float8 tensorwise scaled training active")

    def convert(self, model: nn.Module):
        """
        This function converts the linear layers of `model` to `Float8Linear`.
        Note that today, only dynamic tensor scaling (the default) is supported.
        This will mutate the model inplace.
        """
        if not self.enabled:
            return

        from torchao.float8 import convert_to_float8_training

        # Mutates the model inplace replacing instances of nn.Linear with Float8Linear
        convert_to_float8_training(
            model,
            config=self.config,
            module_filter_fn=partial(module_filter_fn, filter_fqns=self.filter_fqns),
        )
        print(
            "Swapped to Float8Linear layers with enable_fsdp_float8_all_gather="
            f"{self.config.enable_fsdp_float8_all_gather}"
        )

    def post_optimizer_hook(self, model: Union[nn.Module, list[nn.Module]]):
        if not self.enabled:
            return

        if not self.precompute_scale:
            return

        from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

        models = [model] if isinstance(model, nn.Module) else model
        for m in models:
            precompute_float8_dynamic_scale_for_fsdp(m)
