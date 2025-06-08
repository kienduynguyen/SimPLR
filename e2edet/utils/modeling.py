from typing import Optional, List, Set

import torch


def get_layer_id(layer_name, num_layers):
    if "net.pos_embed" in layer_name:
        return 0
    elif "net.patch_embed" in layer_name:
        return 0
    elif "net.blocks." in layer_name:
        layer_id = int(layer_name[layer_name.find("net.blocks.") :].split(".")[2])
        return layer_id + 1

    return num_layers - 1


norm_module_types = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)


def get_parameters(
    model: torch.nn.Module,
    lr_multi: Optional[float] = 1.0,
    lr_module: Optional[List[str]] = [],
    apply_wd: Optional[bool] = False,
    module_except: Optional[List[str]] = [],
):
    param_group_no_decay = {"params": []}
    param_group_lr_multi = {"params": []}
    param_group_others = {"params": []}

    for module_name, module in model.named_modules():
        if any(nd in module_name for nd in module_except):
            continue

        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if (
                isinstance(module, norm_module_types)
                or isinstance(module, torch.nn.Embedding)
                or "relative_position_bias_table" in module_name
                or "absolute_pos_embed" in module_name
                or param.ndim == 1
            ):
                print("no decay:", module_name)
                param_group_no_decay["params"].append(param)
            elif any(nd in param_name for nd in lr_module):
                print("lr_multi:", module_name)
                param_group_lr_multi["params"].append(param)
            else:
                param_group_others["params"].append(param)

    if lr_multi is not None and lr_multi != 1.0:
        param_group_lr_multi["lr_multi"] = lr_multi

    if not apply_wd:
        param_group_no_decay["weight_decay"] = 0.0

    optimizer_grouped_parameters = [
        param_group_no_decay,
        param_group_lr_multi,
        param_group_others,
    ]

    return optimizer_grouped_parameters


def get_vit_parameters(
    model: torch.nn.Module,
    apply_wd: Optional[bool] = False,
    wd_except: Optional[List[str]] = None,
    lr_decay_rate: Optional[float] = None,
    num_layers: Optional[int] = None,
):
    memo: Set[torch.nn.parameter.Parameter] = set()

    if lr_decay_rate is not None:
        assert num_layers is not None
        num_layers += 2

    if lr_decay_rate is not None:
        param_group_decay = [{"params": []} for _ in range(num_layers + 1)]
        param_group_no_decay = [
            {"params": [], "weight_decay": 0.0} for _ in range(num_layers + 1)
        ]
    else:
        param_group_decay = [{"params": []}]
        param_group_no_decay = [{"params": [], "weight_decay": 0.0}]

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if param in memo:
                continue
            memo.add(param)

            no_decay = False
            if not apply_wd and (
                isinstance(module, norm_module_types)
                or param.ndim == 1
                or any(nd in param_name for nd in wd_except)
            ):
                no_decay = True

            if lr_decay_rate is not None:
                layer_id = get_layer_id(f"{module_name}.{param_name}", num_layers)
                if no_decay:
                    param_group_no_decay[layer_id]["params"].append(param)
                    param_group_no_decay[layer_id]["lr_multi"] = lr_decay_rate ** (
                        num_layers - 1 - layer_id
                    )
                else:
                    param_group_decay[layer_id]["params"].append(param)
                    param_group_decay[layer_id]["lr_multi"] = lr_decay_rate ** (
                        num_layers - 1 - layer_id
                    )
            else:
                if no_decay:
                    param_group_no_decay[0]["params"].append(param)
                else:
                    param_group_decay[0]["params"].append(param)
    optimizer_grouped_parameters = param_group_decay + param_group_no_decay

    return optimizer_grouped_parameters
