# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Union, Type

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
from torch.distributed._tensor import distribute_tensor, DTensor
from torch.distributed.checkpoint.state_dict import _init_optim_state
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.optim import Optimizer

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


def _dummy_reset_params(x: nn.Module) -> None:
    """
    Dummy method for patching no-op reset_parameters() when using
    FSDP with meta device.
    """
    return


def prepare_model_for_fsdp_with_meta_device(model: nn.Module) -> nn.Module:
    """
    Dynamically define reset_parameters on every submodule of the model. For LoRA models,
    ensure that the FSDP contract of reset_parameters only modifying a module's directly-owned
    parameters is satisfied. More details here: https://github.com/pytorch/pytorch/issues/104187.

    Args:
        model (nn.Module): model class to prepare for usage with FSDP and meta device.

    Returns:
        nn.Module: Model with reset_parameters defined on every submodule.
        In the case of a LoRA model, we override the default reset_parameters of nn.Linear.

    Raises:
        RuntimeError: if model contains submodule with non-callable attribute reset_parameters
    """
    for k, v in model.named_modules():
        # If the module does not have reset_parameters defined, we define
        # a no-op reset_parameters method to satisfy FSDP's contract.
        reset_params = getattr(v, "reset_parameters", None)

        if reset_params is not None and not callable(reset_params):
            raise RuntimeError(
                f"Cannot override existing reset_parameters variable for FSDP init in {k}"
            )

        if reset_params is None:
            v.reset_parameters = _dummy_reset_params.__get__(v)

    return model


def load_from_full_model_state_dict(
    model,  # noqa
    full_sd: Dict[str, Any],
    device: torch.device,
    strict: bool = False,
    cpu_offload: bool = False,
):
    """
    Converting full state dict into a sharded state dict
    and loading it into FSDP model
    - 'full' means plain tensor
    - 'sharded' means `DTensor` where reach rank has a shard of the plain tensor
    """
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    for param_name, full_tensor in full_sd.items():
        sharded_meta_param = meta_sharded_sd.get(param_name)
        full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)
        sharded_tensor = distribute_tensor(
            full_tensor,
            sharded_meta_param.device_mesh,
            sharded_meta_param.placements,
        )
        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    # choose `assign=True` since we cannot call `copy_` on meta tensor
    return model.load_state_dict(sharded_sd, strict=strict, assign=True)


def get_full_model_state_dict(
    model,  # noqa
    is_rank_zero: bool,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Converting sharded state dict into a full state dict on CPU
    Returning non-empty result on rank0 to avoid peaking CPU memory

    Args:
        model (FSDPModule): wrapped module
        is_rank_zero (bool): flag to check if the process is on rank 0
        device (Optional[torch.device]): device to use for sharded tensors. Default: None

    Raises:
        AssertionError: if the model contains NF4Tensor and the model is not wrapped with FSDP

    Returns:
        Dict[str, Any]: State dict on CPU
    """
    # [Warning] FSDPModel.state_dict converts all Parameter Tensors to DTensors
    sharded_sd = model.state_dict()
    cpu_state_dict = {}
    for param_name, sharded_param in sharded_sd.items():
        if sharded_param.is_cpu:
            assert device is not None and device.type == "cuda", (
                f"Expect cuda but got device={device}. "
                "Please call get_full_model_state_dict(..., device=self._device),"
                " so DTensor can communicate over NCCL."
            )
            sharded_param = sharded_param.to(device)
        full_param = sharded_param.full_tensor()
        if is_rank_zero:
            cpu_state_dict[param_name] = full_param.cpu()
        else:
            del full_param
    return cpu_state_dict


def get_full_optimizer_state_dict(
    opt: Optimizer,
    is_rank_zero: bool,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Converting optimizer state from sharded to full
    For example, "exp_avg" in AdamW is `DTensor`,
    "exp_avg.full_tensor()" converts it to plain tensor on rank 0
    Returning non-empty cpu state dict on rank 0
    """
    sharded_sd = opt.state_dict()
    sharded_state = sharded_sd["state"]
    full_state = {}
    for group_id, sharded_group in sharded_state.items():
        group_state = {}
        for attr, sharded_tensor in sharded_group.items():
            # "exp_avg" in AdamW is `DTensor`
            if isinstance(sharded_tensor, DTensor):
                if sharded_tensor.is_cpu:
                    assert device is not None and device.type == "cuda", (
                        f"Expect cuda but got device={device}. "
                        "Please call get_full_optimizer_state_dict(..., device=self._device),"
                        " so DTensor can communicate over NCCL."
                    )
                    sharded_tensor = sharded_tensor.to(device)
                full_tensor = sharded_tensor.full_tensor()
            else:
                # "step" in AdamW is plain tensor
                full_tensor = sharded_tensor
            if is_rank_zero:
                group_state[attr] = full_tensor.cpu()
            else:
                del full_tensor
        if is_rank_zero:
            full_state[group_id] = group_state
        else:
            del group_state
    if is_rank_zero:
        return {
            "param_groups": sharded_sd["param_groups"],
            "state": full_state,
        }
    else:
        return {}


def load_from_full_optimizer_state_dict(
    opt: Optimizer,
    full_sd: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Converting full optimizer state to sharded state dict
    and loading it into optimizer
    """
    PARAMS = "params"  # noqa: N806
    _init_optim_state(opt)
    param_groups = opt.state_dict()["param_groups"]
    state = opt.state_dict()["state"]

    full_param_groups = full_sd["param_groups"]
    full_state = full_sd["state"]

    for param_group, full_param_group in zip(param_groups, full_param_groups):
        for key, value in full_param_group.items():
            if key == PARAMS:
                continue
            param_group[key] = value
        for pid, full_pid in zip(param_group[PARAMS], full_param_group[PARAMS]):
            if pid not in state:
                continue
            param_state = state[pid]
            full_param_state = full_state[full_pid]
            for attr, full_tensor in full_param_state.items():
                sharded_tensor = param_state[attr]
                if isinstance(sharded_tensor, DTensor):
                    # exp_avg is DTensor
                    param_state[attr] = distribute_tensor(
                        full_tensor,
                        sharded_tensor.device_mesh,
                        sharded_tensor.placements,
                    )
                else:
                    # step is plain tensor
                    param_state[attr] = full_tensor
    opt.load_state_dict(
        {
            "param_groups": param_groups,
            "state": state,
        }
    )


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
        reshard_after_forward = int(layer_id) < len(model.layers) - 1
    else:
        raise ValueError(
            f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
        )

    if model._contain_connector:
        for layer_id, transformer_block in model.connector.blocks.named_children():
            reshard_after_forward = int(layer_id) < len(model.connector.blocks) - 1
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        fully_shard(model.connector, **fsdp_config, reshard_after_forward=True)

    fully_shard(model.llm.model.embed_tokens, **fsdp_config, reshard_after_forward=True)
    fully_shard(model.llm.lm_head, **fsdp_config, reshard_after_forward=True)

    for sharded_module in model.llm._block_shard:
        try:
            submodule = model.llm.get_submodule(sharded_module)
            for layer_id, transformer_block in submodule.named_children():
                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=True,
                )
        except AttributeError:
            continue
    fully_shard(model.llm, **fsdp_config, reshard_after_forward=True)

    if model.vision_backbone._block_shard is not None:
        backbone_layers = model.vision_backbone.get_submodule(
            model.vision_backbone._block_shard
        )

        for transformer_block in backbone_layers.children():
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=True,
            )
    fully_shard(model.vision_backbone, **fsdp_config, reshard_after_forward=True)
    fully_shard(model, **fsdp_config, reshard_after_forward=True)


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

    for compiled_module in model.llm._block_compile:
        try:
            submodule = model.llm.get_submodule(compiled_module)

            for layer_id, transformer_block in submodule.named_children():
                transformer_block = torch.compile(transformer_block, fullgraph=True)
                submodule.register_module(layer_id, transformer_block)
        except AttributeError:
            continue

    if model.vision_backbone._block_compile is not None:
        backbone_layers = model.vision_backbone.get_submodule(
            model.vision_backbone._block_compile
        )

        for layer_id, transformer_block in backbone_layers.named_children():
            transformer_block = torch.compile(transformer_block, fullgraph=True)
            backbone_layers.register_module(layer_id, transformer_block)

    try:
        connector_blocks = model.get_submodule("connector.blocks")
    except AttributeError:
        connector_blocks = None

    if connector_blocks is not None:
        for layer_id, transformer_block in connector_blocks.named_children():
            transformer_block = torch.compile(transformer_block, fullgraph=True)
            connector_blocks.register_module(layer_id, transformer_block)

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

    for ac_module in model.llm._block_ac:
        try:
            submodule = model.llm.get_submodule(ac_module)

            for layer_id, transformer_block in submodule.named_children():
                transformer_block = _apply_ac_to_transformer_block(
                    transformer_block, mode, selective_ac_option
                )
                submodule.register_module(layer_id, transformer_block)
        except AttributeError:
            continue

    if model.vision_backbone._block_ac is not None:
        backbone_layers = model.vision_backbone.get_submodule(
            model.vision_backbone._block_ac
        )

        for layer_id, transformer_block in backbone_layers.named_children():
            transformer_block = _apply_ac_to_transformer_block(
                transformer_block, mode, selective_ac_option
            )
            backbone_layers.register_module(layer_id, transformer_block)

    try:
        connector_blocks = model.get_submodule("connector.blocks")
    except AttributeError:
        connector_blocks = None

    if connector_blocks is not None:
        for layer_id, transformer_block in connector_blocks.named_children():
            transformer_block = _apply_ac_to_transformer_block(
                transformer_block, mode, selective_ac_option
            )
            connector_blocks.register_module(layer_id, transformer_block)

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
    for transformer_block in model.layers.values():
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
                tp_enabled=parallel_dims.tp_enabled,
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
