import importlib
import collections.abc
import os
import copy

import torch
import torch.optim as optim
import omegaconf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from e2edet.utils.general import get_optimizer_parameters, contains_fsdp

from .optim_wrapper import OptimizersContainer, OptimizersInBackwardContainer


OPTIM_REGISTRY = {"adam": optim.Adam, "adamw": optim.AdamW}


def build_optimizer(config, model):
    optim_type = config.optimizer["type"]
    optim_config = copy.deepcopy(config.optimizer["params"])

    with omegaconf.open_dict(optim_config):
        redundants = ["lr_decay_rate", "wd_norm", "wd_bias"]
        for redundant in redundants:
            if redundant in optim_config:
                optim_config.pop(redundant)

        early_step_in_backward = optim_config.pop("early_step_in_backward", False)

    if optim_type not in OPTIM_REGISTRY:
        raise ValueError("Optimizer ({}) is not found.".format(optim_type))

    model_params = get_optimizer_parameters(model)

    if isinstance(model_params[0], collections.abc.Sequence):
        param_groups = []
        backbone_group, transformer_group = model_params

        with omegaconf.open_dict(optim_config):
            lr_backbone = optim_config.pop("lr_backbone", optim_config["lr"])

            for bgroup in backbone_group:
                if "lr_multi" in bgroup:
                    bgroup["lr"] = lr_backbone * bgroup.pop("lr_multi")
                else:
                    bgroup["lr"] = lr_backbone
                param_groups.append(bgroup)

            for tgroup in transformer_group:
                if "lr_multi" in tgroup:
                    tgroup["lr"] = optim_config["lr"] * tgroup.pop("lr_multi")
                param_groups.append(tgroup)
    elif isinstance(model_params[0], collections.abc.Mapping):
        param_groups = model_params
    else:
        param_groups = [{"lr": optim_config["lr"], "params": model_params}]

    if contains_fsdp(model):
        optimizer_cls = OPTIM_REGISTRY[optim_type]
        if early_step_in_backward:
            optimizer = OptimizersInBackwardContainer(
                model, param_groups, optimizer_cls, optim_config
            )
        else:
            optimizer = OptimizersContainer(
                model, param_groups, optimizer_cls, optim_config
            )
    else:
        optimizer = OPTIM_REGISTRY[optim_type](param_groups, **optim_config)

    return optimizer


def register_optim(name):
    def register_optim_cls(cls):
        if name in OPTIM_REGISTRY:
            raise ValueError("Cannot register duplicate optimizer ({})".format(name))
        elif not issubclass(cls, torch.optim.Optimizer):
            raise ValueError(
                "Optimizer ({}: {}) must extend torch.optim.Optimizer".format(
                    name, cls.__name__
                )
            )

        OPTIM_REGISTRY[name] = cls
        return cls

    return register_optim_cls


optims_dir = os.path.dirname(__file__)
for file in os.listdir(optims_dir):
    path = os.path.join(optims_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        optim_name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("e2edet.optim." + optim_name)
