# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Callable, Union, Type

import torch
from torch import nn

from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed._composable.replicate import replicate
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from e2edet.utils.logger import logger

from .precision import get_dtype
from .parallel_dims import ParallelDims


FSDPPolicyType: Type = Callable[[nn.Module, bool, int], bool]

FSDPPolicyType.__doc__ = """

A datatype for a function that can be used as an FSDP wrapping policy.
In particular, this type denotes a function that can accept an nn.Module, a boolean flag, and an integer
and return a boolean indicating whether the module should be wrapped with FSDP. Objects of this type can
be directly passed into PyTorch FSDP's ``auto_wrap_policy`` argument to specify how FSDP wraps submodules.

The below function serves as an example of creating and returning a function that obeys the contract of
``FSDPPolicyType``::

    def get_fsdp_policy(module: nn.Module, modules_to_wrap: Set[Type], min_num_params: int):

        def my_fsdp_policy(module: nn.Module, modules_to_wrap: Set[Type], recurse: bool, min_num_params: int) -> bool:
            if recurse:
                return True
            # Wrap layers that are of type in ``modules_to_wrap`` and layers with more than min_num_params

            return isinstance(module, tuple(modules_to_wrap)) or sum(p.numel() for p in module.parameters()) > 1000

        return functools.partial(my_fsdp_policy, modules_to_wrap=modules_to_wrap)

Please see documentation of ``auto_wrap_policy`` at https://pytorch.org/docs/stable/fsdp.html for additional details.

"""

# for selective op activation checkpointing
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
}


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    for module in model.shard_modules():
        blocks = model.get_submodule(module)

        for block_id, block in blocks.named_children():
            if reshard_after_forward_policy == "always":
                reshard_after_forward = True
            elif reshard_after_forward_policy == "never":
                reshard_after_forward = False
            elif reshard_after_forward_policy == "default":
                # if pp_enabled:
                #     # For PP, do not reshard after forward to avoid per-microbatch
                #     # all-gathers, which can be expensive and non-overlapped
                #     reshard_after_forward = False
                # else:
                #     # As an optimization, do not reshard after forward for the last
                #     # transformer block since FSDP would prefetch it immediately
                #     reshard_after_forward = int(layer_id) < len(model.layers) - 1
                reshard_after_forward = int(block_id) < len(blocks) - 1
            else:
                raise ValueError(
                    f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
                )
            fully_shard(
                block, **fsdp_config, reshard_after_forward=reshard_after_forward
            )
    fully_shard(model, **fsdp_config, reshard_after_forward=False)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")


def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """

    for module in model.shard_modules():
        blocks = model.get_submodule(module)

        for block_id, block in blocks.named_children():
            block = torch.compile(block)
            blocks.register_module(block_id, block)

    # for module_id in model.compile_modules():
    #     module = model.get_submodule(module_id)
    #     module = torch.compile(module)
    #     model.register_module(module_id, module)

    logger.info("Compiling each TransformerBlock with torch.compile")


def _apply_ac_to_transformer_block(
    module: nn.Module, mode: str, selective_ac_option: Union[str, int]
):
    valid_ac_modes = ("full", "selective")
    if mode not in valid_ac_modes:
        raise ValueError(f"Invalid AC mode: {mode}. Valid modes: {valid_ac_modes}")

    if mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert mode == "selective", f"{mode}"
    use_op_sac = selective_ac_option == "op"
    use_layer_sac = selective_ac_option
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )
    if use_op_sac:
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in _save_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
    elif use_layer_sac:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ac_freq = int(selective_ac_option)
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            return module


def apply_ac(model: nn.Module, mode: str, selective_ac_option: Union[int, str]):
    """Apply activation checkpointing to the model."""
    for module in model.shard_modules():
        blocks = model.get_submodule(module)

        for block_id, block in blocks.named_children():
            block = _apply_ac_to_transformer_block(block, mode, selective_ac_option)
            block.register_module(block_id, block)

    logger.info(f"Applied {mode} activation checkpointing to the model")


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for module in model.shard_modules():
        blocks = model.get_submodule(module)

        for transformer_block in blocks.values():
            layer_plan = {
                "attention_norm": SequenceParallel(),
                "attention": prepare_module_input(
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.wq": colwise_parallel(),
                "attention.wk": colwise_parallel(),
                "attention.wv": colwise_parallel(),
                "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
                "ffn_norm": SequenceParallel(),
                "feed_forward": prepare_module_input(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": colwise_parallel(),
                "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                "feed_forward.w3": colwise_parallel(),
            }

            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )


def parallelize_model(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    use_compile: bool,
    enable_async_tp: bool,
    ac_mode: str,
    selective_ac_option: Union[int, str],
    mp_param: torch.dtype,
    mp_reduce: torch.dtype,
    enable_compiled_autograd: bool,
    cpu_offload: bool = False,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    if parallel_dims.tp_enabled:
        if enable_async_tp and not use_compile:
            raise RuntimeError("Async TP requires use_compile=True")
        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
            enable_async_tp=enable_async_tp,
        )

    if ac_mode != "none":
        apply_ac(model, ac_mode, selective_ac_option)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if use_compile:
        apply_compile(model)

    if parallel_dims.dp_enabled:
        if parallel_dims.dp_shard_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mesh = world_mesh["dp_replicate", "dp_shard"]
            else:
                dp_mesh = world_mesh["dp"]

            apply_fsdp(
                model,
                dp_mesh,
                param_dtype=get_dtype(mp_param),
                reduce_dtype=get_dtype(mp_reduce),
                cpu_offload=cpu_offload,
                # tp_enabled=parallel_dims.tp_enabled,
            )
            if parallel_dims.dp_replicate_enabled:
                logger.info("Applied HSDP to the model")
            else:
                logger.info("Applied FSDP to the model")
        else:
            if world_mesh.ndim > 1:
                raise RuntimeError("DDP has not supported > 1D parallelism")
            apply_ddp(
                model,
                world_mesh,
                enable_compile=use_compile,
                enable_compiled_autograd=enable_compiled_autograd,
            )
