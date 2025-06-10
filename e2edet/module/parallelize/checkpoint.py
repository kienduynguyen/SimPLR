import os
import glob
import contextlib
import re
import shutil
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Iterable,
)
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    StateDictOptions,
    _verify_state_dict,
    _unflatten_model_state_dict,
    _verify_options,
    _gc_context,
    _iterate_valid_model_state,
    _get_fqns,
    _broadcast_state_dict,
    _distribute_state_dict,
    _state_dict_fn,
    _load_optim_state_dict,
)
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from omegaconf import OmegaConf

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

from e2edet.utils.general import get_root, GarbageCollection
from e2edet.utils.distributed import is_master, synchronize
from e2edet.utils.logger import logger, log_once


FQNS_T = Set[str]
PrimitiveType = Union[DTensor, ShardedTensor, torch.Tensor, int, float, str]
ValueType = Union[
    PrimitiveType, List[PrimitiveType], Tuple[PrimitiveType], Dict[str, "ValueType"]
]
DictValueType = Dict[str, ValueType]
ListDictValueType = List[DictValueType]
OptimizerStateType = Dict[str, Union[DictValueType, ListDictValueType]]


@dataclass
class _StateDictInfo(StateDictOptions):
    fqn_param_mapping: Dict[Union[str, torch.Tensor], Union[FQNS_T, torch.Tensor]] = (
        field(default_factory=dict)
    )
    shared_params_mapping: Dict[
        Union[str, torch.Tensor], Union[FQNS_T, torch.Tensor]
    ] = field(default_factory=dict)
    submodule_prefixes: Set[str] = field(default_factory=set)
    handle_model: bool = True
    handle_optim: bool = True
    fsdp_context: Callable = contextlib.nullcontext
    fsdp_modules: List[nn.Module] = field(default_factory=list)


def set_model_state_dict(
    model: nn.Module,
    model_state_dict: Dict[str, ValueType],
    *,
    options: Optional[StateDictOptions] = None,
) -> _IncompatibleKeys:
    """Load the model state_dict.

    The counterpart of ``get_model_state_dict`` to set the state_dict to the
    model. See ``set_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        model_state_dict: (Dict[str, ValueType]):
           the model state_dict to load. If the key of the ``model_state_dict``
           is nn.Module, the key is a submodule of ``model`` and the value should
           be the state_dict of the submodule. When loading the state_dict,
           the prefix of the submodule will be append to the state_dict.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    :type model_state_dict: typing.Dict[str, ValueType]
    """
    model_state_dict: Dict[str, ValueType] = _unflatten_model_state_dict(
        model, model_state_dict
    )
    with _gc_context():
        info = _verify_options(model, (), optim_only=False, options=options)

        _verify_state_dict(model_state_dict, {}, info)
        return _load_model_state_dict(model, model_state_dict, info)


@torch.no_grad()
def _load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, ValueType],
    info: _StateDictInfo,
) -> _IncompatibleKeys:
    if not info.handle_model or (not state_dict and not info.broadcast_from_rank0):
        return _IncompatibleKeys({}, {})

    local_state_dict = {}
    for key, value in _iterate_valid_model_state(model):
        if info.ignore_frozen_params and not value.requires_grad:
            continue

        fqns = _get_fqns(model, key)
        fqns_with_prefix = _get_fqns(
            model, key, skip_ddp_prefix=False, skip_compiler_prefix=False
        )

        for fqn, fqn_with_prefix in zip(fqns, fqns_with_prefix):
            if (
                not info.broadcast_from_rank0 or dist.get_rank() == 0
            ) and fqn != fqn_with_prefix:
                state_dict[fqn_with_prefix] = state_dict.pop(fqn)
            local_state_dict[fqn_with_prefix] = value

    assign = False
    if info.broadcast_from_rank0 or info.full_state_dict:
        device = None
        for key, value in local_state_dict.items():
            if torch.is_tensor(value) and value.dim() > 0:
                if device is None:
                    device = value.device
                else:
                    assert device == value.device
        assert device is not None
        if device == torch.device("meta"):
            device = dist.distributed_c10d._get_pg_default_device()
            assign = True
        if info.broadcast_from_rank0:
            _broadcast_state_dict(
                state_dict, local_state_dict, device=device, strict=info.strict
            )
        elif info.full_state_dict:
            _distribute_state_dict(state_dict, local_state_dict, device=device)
        for fqn, local_state in local_state_dict.items():
            state_dict[fqn] = local_state

    with info.fsdp_context():
        return cast(
            _IncompatibleKeys,
            _state_dict_fn(model, "load_state_dict")(
                state_dict=state_dict, strict=info.strict, assign=assign
            ),
        )


def set_optimizer_state_dict(
    model: nn.Module,
    optimizers: Union[torch.optim.Optimizer, Iterable[torch.optim.Optimizer]],
    optim_state_dict: OptimizerStateType,
    *,
    options: Optional[StateDictOptions] = None,
) -> None:
    """Load the optimizers state_dict.

    The counterpart of ``get_optimizer_state_dict`` to set the state_dict to the
    optimizers. See ``set_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        optim_state_dict: OptimizerStateType:
            the optimizer state_dict to load.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        None

    :type optim_state_dict: typing.OptimizerStateType
    """
    with _gc_context():
        optimizers = (
            (optimizers,)
            if isinstance(optimizers, torch.optim.Optimizer)
            else tuple(optimizers)
        )

        orig_defaults = [[] for _ in optimizers]
        for orig_default, optimizer in zip(orig_defaults, optimizers):
            for group in optimizer.param_groups:
                orig_state_dict = {}
                for kk, vv in group.items():
                    if kk in optimizer.defaults:
                        orig_state_dict[kk] = vv
                orig_default.append(orig_state_dict)
        info = _verify_options(model, optimizers, optim_only=True, options=options)

        _verify_state_dict({}, optim_state_dict, info)
        _load_optim_state_dict(model, optimizers, optim_state_dict, info)

        for orig_default, optimizer in zip(orig_defaults, optimizers):
            for orig_state_dict, group in zip(orig_default, optimizer.param_groups):
                for kk, vv in orig_state_dict.items():
                    if kk not in group:
                        group.setdefault(kk, vv)
                        log_once(
                            logger, f"{kk} is not in param_groups, setting to {vv}"
                        )


@torch.no_grad()
def save_with_gc(state, checkpoint_id):
    dcp.save(state, checkpoint_id=checkpoint_id)
    GarbageCollection.collect("GC collection invoked by checkpointer.")


@dataclass
class TrainState(Stateful):
    def __init__(self, trainer):
        self.trainer = trainer

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        step = self.trainer.current_update
        epoch = self.trainer.current_epoch

        state_dict = {
            "step": torch.tensor(step, dtype=torch.int32),
            "epoch": torch.tensor(epoch, dtype=torch.int32),
            "lr_scheduler": self.trainer.lr_scheduler.state_dict(),
        }
        if self.trainer.grad_scaler is not None:
            state_dict["grad_scaler"] = self.trainer.grad_scaler.state_dict()
        print("state_dict:", state_dict)

        return state_dict

    def load_state_dict(self, state_dict) -> None:
        self.trainer.current_update = state_dict["step"].item()
        self.trainer.current_epoch = state_dict["epoch"].item()
        self.trainer.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        if self.trainer.grad_scaler is not None:
            self.trainer.grad_scaler.load_state_dict(state_dict["grad_scaler"])


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def state_dict(self) -> None:
        return get_model_state_dict(
            self.model,
            options=StateDictOptions(ignore_frozen_params=True),
        )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        incompatible_keys = set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(
                strict=False,
                ignore_frozen_params=True,
            ),
        )
        print("Model loaded:", incompatible_keys)


class CheckpointManager:
    def __init__(self, config, trainer):
        self.config = config
        self.save_dir = self.config.training.save_dir
        self.num_checkpoint = self.config.training.num_checkpoint
        self.model_name = self.config.model
        self.device = trainer.device
        self.trainer = trainer

        # self.pg = dist.new_group(backend="gloo")

        self.states = {
            "model": ModelWrapper(trainer.model),
            "optimizer": trainer.optimizer,
            "train_state": TrainState(trainer),
        }

        # if get_local_rank() == 0:
        if is_master():
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)

        self.final_path = os.path.join(self.save_dir, self.model_name + "_final.pth")
        self.models_foldername = os.path.join(self.save_dir, "models")

        # if not os.path.exists(self.models_foldername) and get_local_rank() == 0:
        if not os.path.exists(self.models_foldername) and is_master():
            os.makedirs(self.models_foldername, exist_ok=True)

        logger.info("Saving config...")
        self.save_config()

    def _extract_iter(self, path):
        return int(path.split("_")[-1].split(".")[0])

    def _process_config(self):
        save_config = OmegaConf.create(OmegaConf.to_yaml(self.config, resolve=True))
        save_config.distributed.init_method = None
        save_config.distributed.rank = 0
        save_config.distributed.port = -1
        save_config.distributed.backend = "nccl"
        save_config.distributed.world_size = 1
        save_config.distributed.no_spawn = False

        return save_config

    def save_config(self):
        # if get_local_rank() == 0:
        if is_master():
            cfg_file = os.path.join(self.save_dir, "config.yaml")
            save_config = self._process_config()

            with open(cfg_file, "w") as f:
                f.write(OmegaConf.to_yaml(save_config, resolve=True))

    def load_state_dict(self):
        tp = self.config.training

        if tp.resume:
            ckpt_file_paths = sorted(
                glob.glob(os.path.join(self.models_foldername, "model_*")),
                key=self._extract_iter,
            )

            if len(ckpt_file_paths) > 0:
                ckpt_filepath = ckpt_file_paths[-1]
                logger.info("Loading weights the last checkpoint")
                self._load(ckpt_filepath)

                return True

            if tp.resume_file is not None and tp.resume_dcp:
                logger.info("Loading weights from {}".format(tp.resume_file))

                if os.path.exists(tp.resume_file):
                    self._load(tp.resume_file, model_only=True)
                else:
                    raise RuntimeError("{} doesn't exist".format(tp.resume_file))

        return False

    def _load(self, file, model_only=False):
        if not os.path.isabs(file):
            file = os.path.normpath(os.path.join(get_root(), "..", file))

        if model_only:
            logger.info(f"Loading model only from {file}")
            states = {"model": self.states["model"]}
        else:
            logger.info(f"Loading full checkpoint from {file}")
            states = self.states

        original_stateful_states = {
            k: v for k, v in states.items() if isinstance(v, Stateful)
        }

        dcp.load(
            states,
            checkpoint_id=file,
            planner=DefaultLoadPlanner(allow_partial_load=True),
            # process_group=self.pg,
        )
        states.update(original_stateful_states)
        # self.states["train_state"] = TrainState(self.trainer)
        # self.states["model"] = ModelWrapper(self.trainer.model)
        # self.states["optimizer"] = OptimizerWrapper(
        #     self.trainer.model, self.trainer.optimizer
        # )
        logger.info("Checkpoint loaded")

    def save(self, update=None, model_weights_only=False):
        if model_weights_only:
            state_dict = get_model_state_dict(
                self.trainer.model,
                options=StateDictOptions(
                    ignore_frozen_params=True,
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
            if is_master():
                torch.save(state_dict, self.final_path)
        else:
            assert update is not None
            ckpt_filepath = os.path.join(self.models_foldername, "model_%d" % update)
            ckpt = self.states

            # dcp.save(ckpt, checkpoint_id=ckpt_filepath, process_group=self.pg)
            # dcp.save(ckpt, checkpoint_id=ckpt_filepath)
            save_with_gc(ckpt, checkpoint_id=ckpt_filepath)

            self._purge_stale_checkpoints()

    def _purge_stale_checkpoints(self):
        if self.num_checkpoint > 0:
            discovered_checkpoints = []
            for filename in os.listdir(self.models_foldername):
                match = re.search(r"model_(\d+)", filename)
                path = os.path.join(self.models_foldername, filename)
                discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.num_checkpoint]

            for _, path in to_delete:
                logger.info(f"Deleting old checkpoint {path}")
                shutil.rmtree(path, ignore_errors=True)

    def finalize(self):
        self.save(model_weights_only=True)
        synchronize()
