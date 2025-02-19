import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from e2edet.module import build_matcher
from e2edet.utils import box_ops
from e2edet.utils.distributed import get_world_size, is_dist_avail_and_initialized
from .loss_utils import (
    BaseLoss,
    LabelLoss,
    FocalLabelLoss,
    FedLabelLoss,
    BCELabelLoss,
    Boxes3DLoss,
    BoxesLoss,
    PanopticMaskLoss,
    InstanceMaskLoss,
)


LOSS_REGISTRY = {}


def build_loss(loss_config, num_classes, iter_per_update):
    if loss_config["type"] not in LOSS_REGISTRY:
        raise ValueError("Loss ({}) is not found.".format(loss_config["type"]))

    loss_cls = LOSS_REGISTRY[loss_config["type"]]
    params = loss_config["params"]
    weight_dict = {"loss_ce": params["class_loss_coef"]}

    if loss_config["type"] == "detr":
        weight_dict["loss_bbox"] = params["bbox_loss_coef"]
        weight_dict["loss_giou"] = params["giou_loss_coef"]

        losses = ["boxes", "labels"]
        other_params = {"iter_per_update": iter_per_update}
    elif (
        loss_config["type"] == "universal_boxer2d"
        or loss_config["type"] == "universal_boxer2d_v2"
    ):
        weight_dict["loss_bbox"] = params["bbox_loss_coef"]
        weight_dict["loss_giou"] = params["giou_loss_coef"]

        if params.get("losses", None) is not None:
            losses = params["losses"]
        else:
            losses = ["boxes", "focal_labels", "masks"]
        weight_dict["loss_mask"] = params["mask_loss_coef"]
        weight_dict["loss_dice"] = params["dice_loss_coef"]

        other_params = {
            "num_points": params["num_points"],
            "oversample_ratio": params["oversample_ratio"],
            "importance_sample_ratio": params["importance_sample_ratio"],
            "use_uncertainty": params.get("use_uncertainty", True),
            "iter_per_update": iter_per_update,
        }
        if loss_config["type"] == "universal_boxer2d_v2":
            other_params["dataset_name"] = params["dataset_name"]
            other_params["freq_weight_power"] = params["freq_weight_power"]
            other_params["num_fed_loss_classes"] = params["num_fed_loss_classes"]
    elif loss_config["type"] == "mask2former":
        if params.get("losses", None) is not None:
            losses = params["losses"]
        else:
            losses = ["labels", "masks"]
        weight_dict["loss_mask"] = params["mask_loss_coef"]
        weight_dict["loss_dice"] = params["dice_loss_coef"]

        other_params = {
            "num_points": params["num_points"],
            "oversample_ratio": params["oversample_ratio"],
            "importance_sample_ratio": params["importance_sample_ratio"],
            "use_uncertainty": params.get("use_uncertainty", True),
            "iter_per_update": iter_per_update,
        }
    elif loss_config["type"] == "boxer2d":
        weight_dict["loss_bbox"] = params["bbox_loss_coef"]
        weight_dict["loss_giou"] = params["giou_loss_coef"]

        if params.get("losses", None) is not None:
            losses = params["losses"]
        else:
            losses = ["boxes", "focal_labels"]
        if params["use_mask"]:
            weight_dict["loss_mask"] = params["mask_loss_coef"]
            weight_dict["loss_dice"] = params["dice_loss_coef"]
            losses.append("masks")

        other_params = {
            "instance_mask": params["instance_mask"],
            "iter_per_update": iter_per_update,
        }
    elif loss_config["type"] == "boxer3d":
        weight_dict["loss_bbox"] = params["bbox_loss_coef"]
        weight_dict["loss_giou"] = params["giou_loss_coef"]

        losses = ["boxes", "focal_labels"]
        weight_dict["loss_rad"] = params["rad_loss_coef"]
    else:
        raise ValueError(
            "Only detr|boxer2d|boxer3d|universal_boxer2d|mask2former losses are supported (found {})".format(
                loss_config["type"]
            )
        )

    matcher = build_matcher(params["matcher"])
    module_loss = loss_cls(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        **other_params,
    )

    return module_loss


def register_loss(name):
    def register_loss_cls(cls):
        if name in LOSS_REGISTRY:
            raise ValueError("Cannot register duplicate loss ({})".format(name))

        LOSS_REGISTRY[name] = cls
        return cls

    return register_loss_cls


@register_loss("detr")
class DETRLoss(BaseLoss):
    def __init__(
        self, num_classes, matcher, weight_dict, eos_coef, losses, iter_per_update
    ):
        defaults = dict(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
            losses=losses,
            iter_per_update=iter_per_update,
        )
        super().__init__("detr", defaults)

        self.detr_losses = nn.ModuleDict()
        for loss in losses:
            if loss == "boxes":
                self.detr_losses[loss] = BoxesLoss()
            elif loss == "labels":
                self.detr_losses[loss] = LabelLoss(
                    num_classes, eos_coef, iter_per_update
                )
            else:
                raise ValueError(
                    "Only boxes|labels|balanced_labels are supported for detr "
                    "losses. Found {}".format(loss)
                )

    def get_target_classes(self):
        for kk in self.detr_losses.keys():
            if "labels" in kk:
                return (
                    self.detr_losses[kk].src_logits,
                    self.detr_losses[kk].target_classes,
                )

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if "num_boxes" not in outputs:
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor(
                [num_boxes],
                dtype=torch.float,
                device=next(iter(outputs.values())).device,
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            num_boxes = outputs["num_boxes"].item()

        # Compute all the requested losses
        losses = {}

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.detr_losses[loss](
                        aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        for loss in self.losses:
            losses.update(self.detr_losses[loss](outputs, targets, indices, num_boxes))

        return losses


@register_loss("mask2former")
class Mask2FormerLoss(BaseLoss):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        use_uncertainty,
        iter_per_update,
    ):
        defaults = dict(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            use_uncertainty=use_uncertainty,
            iter_per_update=iter_per_update,
        )
        super().__init__("mask2former", defaults)

        self.boxer2d_losses = nn.ModuleDict()
        self.boxer2d_enc_losses = nn.ModuleDict()
        for loss in losses:
            if loss == "boxes":
                self.boxer2d_losses[loss] = BoxesLoss()
                self.boxer2d_enc_losses[loss + "_enc"] = BoxesLoss()
            elif loss == "focal_labels":
                self.boxer2d_losses[loss] = FocalLabelLoss(num_classes, 0.25)
                self.boxer2d_enc_losses[loss + "_enc"] = FocalLabelLoss(1, 0.25)
            elif loss == "labels":
                self.boxer2d_losses[loss] = LabelLoss(num_classes, 0.1, iter_per_update)
                self.boxer2d_enc_losses[loss + "_enc"] = LabelLoss(
                    1, 0.1, iter_per_update
                )
            elif loss == "masks":
                focal_label = "focal_labels" in losses
                self.boxer2d_losses[loss] = PanopticMaskLoss(
                    num_points,
                    oversample_ratio,
                    importance_sample_ratio,
                    focal_label=focal_label,
                    use_uncertainty=use_uncertainty,
                )
                self.boxer2d_enc_losses[loss + "_enc"] = PanopticMaskLoss(
                    num_points,
                    oversample_ratio,
                    importance_sample_ratio,
                    focal_label=focal_label,
                    use_uncertainty=use_uncertainty,
                )
            else:
                raise ValueError(
                    "Only boxes|focal_labels|masks are supported for universal_boxer2d "
                    "losses. Found {}".format(loss)
                )

    def get_target_classes(self):
        for kk in self.boxer2d_losses.keys():
            if "labels" in kk:
                return (
                    self.boxer2d_losses[kk].src_logits,
                    self.boxer2d_losses[kk].target_classes,
                )

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if "num_boxes" not in outputs:
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor(
                [num_boxes],
                dtype=torch.float,
                device=next(iter(outputs.values())).device,
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            num_boxes = outputs["num_boxes"].item()

        # Compute all the requested losses
        losses = {}

        if "enc_outputs" in outputs:
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])

            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                indices = self.matcher(enc_outputs, bin_targets)
                for loss in self.losses:
                    if loss == "masks" and "feat_masks" not in enc_outputs:
                        continue
                    l_dict = self.boxer2d_enc_losses[loss + "_enc"](
                        enc_outputs, bin_targets, indices, num_boxes
                    )
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.boxer2d_losses[loss](
                        aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        for loss in self.losses:
            losses.update(
                self.boxer2d_losses[loss](outputs, targets, indices, num_boxes)
            )

        return losses


@register_loss("universal_boxer2d")
class UniversalBoxer2DLoss(BaseLoss):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        use_uncertainty,
        iter_per_update,
    ):
        defaults = dict(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            use_uncertainty=use_uncertainty,
            iter_per_update=iter_per_update,
        )
        super().__init__("universal_boxer2d", defaults)

        self.boxer2d_losses = nn.ModuleDict()
        self.boxer2d_enc_losses = nn.ModuleDict()
        for loss in losses:
            if loss == "boxes":
                self.boxer2d_losses[loss] = BoxesLoss()
                self.boxer2d_enc_losses[loss + "_enc"] = BoxesLoss()
            elif loss == "focal_labels":
                self.boxer2d_losses[loss] = FocalLabelLoss(num_classes, 0.25)
                self.boxer2d_enc_losses[loss + "_enc"] = FocalLabelLoss(1, 0.25)
            elif loss == "labels":
                self.boxer2d_losses[loss] = LabelLoss(num_classes, 0.1, iter_per_update)
                self.boxer2d_enc_losses[loss + "_enc"] = LabelLoss(
                    1, 0.1, iter_per_update
                )
            elif loss == "bce_labels":
                self.boxer2d_losses[loss] = BCELabelLoss(num_classes)
                self.boxer2d_enc_losses[loss + "_enc"] = BCELabelLoss(1)
            elif loss == "masks":
                focal_label = "focal_labels" in losses
                self.boxer2d_losses[loss] = PanopticMaskLoss(
                    num_points,
                    oversample_ratio,
                    importance_sample_ratio,
                    focal_label=focal_label,
                    use_uncertainty=use_uncertainty,
                )
                self.boxer2d_enc_losses[loss + "_enc"] = PanopticMaskLoss(
                    num_points,
                    oversample_ratio,
                    importance_sample_ratio,
                    focal_label=focal_label,
                    use_uncertainty=use_uncertainty,
                )
            else:
                raise ValueError(
                    "Only boxes|focal_labels|masks are supported for universal_boxer2d "
                    "losses. Found {}".format(loss)
                )

    def get_target_classes(self):
        for kk in self.boxer2d_losses.keys():
            if "labels" in kk:
                return (
                    self.boxer2d_losses[kk].src_logits,
                    self.boxer2d_losses[kk].target_classes,
                )

    def prep_for_dn(self, mask_dict):
        known_indices = mask_dict["known_indices"]
        scalar, pad_size = mask_dict["scalar"], mask_dict["pad_size"]
        assert pad_size % scalar == 0
        single_pad = pad_size // scalar

        num_tgt = known_indices.numel()

        return num_tgt, single_pad, scalar

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        current_device = next(iter(outputs.values())).device

        if "dn_outputs" in outputs:
            mask_dict = outputs.pop("mask_dict")
            exc_idx = []
            if mask_dict is not None:
                num_tgt, single_pad, scalar = self.prep_for_dn(mask_dict)
            else:
                scalar = 1

            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(
                        0,
                        len(targets[i]["labels"]),
                        dtype=torch.long,
                        device=current_device,
                    )
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).to(
                        dtype=torch.long, device=current_device
                    ).unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor(
                        [], dtype=torch.long, device=current_device
                    )

                exc_idx.append((output_idx, tgt_idx))

        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs" and k != "dn_outputs"
        }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if "num_boxes" not in outputs:
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor(
                [num_boxes],
                dtype=torch.float,
                device=current_device,
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            num_boxes = outputs["num_boxes"].item()

        # Compute all the requested losses
        losses = {}

        if "enc_outputs" in outputs:
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])

            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                indices = self.matcher(enc_outputs, bin_targets)
                for loss in self.losses:
                    if loss == "masks" and "pred_masks" not in enc_outputs:
                        continue
                    l_dict = self.boxer2d_enc_losses[loss + "_enc"](
                        enc_outputs, bin_targets, indices, num_boxes
                    )
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.boxer2d_losses[loss](
                        aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "dn_outputs" in outputs:
            for i, dn_outputs in enumerate(outputs["dn_outputs"]):
                for loss in self.losses:
                    l_dict = self.boxer2d_losses[loss](
                        dn_outputs, targets, exc_idx, num_boxes * scalar
                    )
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        for loss in self.losses:
            losses.update(
                self.boxer2d_losses[loss](outputs, targets, indices, num_boxes)
            )

        return losses


@register_loss("universal_boxer2d_v2")
class UniversalBoxer2DLossV2(BaseLoss):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        use_uncertainty,
        dataset_name,
        freq_weight_power,
        num_fed_loss_classes,
        iter_per_update,
    ):
        defaults = dict(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            dataset_name=dataset_name,
            freq_weight_power=freq_weight_power,
            num_fed_loss_classes=num_fed_loss_classes,
            use_uncertainty=use_uncertainty,
            iter_per_update=iter_per_update,
        )
        super().__init__("universal_boxer2d_v2", defaults)

        self.boxer2d_losses = nn.ModuleDict()
        self.boxer2d_enc_losses = nn.ModuleDict()
        for loss in losses:
            if loss == "boxes":
                self.boxer2d_losses[loss] = BoxesLoss()
                self.boxer2d_enc_losses[loss + "_enc"] = BoxesLoss()
            elif loss == "focal_labels":
                self.boxer2d_losses[loss] = FocalLabelLoss(num_classes, 0.25)
                self.boxer2d_enc_losses[loss + "_enc"] = FocalLabelLoss(1, 0.25)
            elif loss == "labels":
                self.boxer2d_losses[loss] = LabelLoss(num_classes, 0.1, iter_per_update)
                self.boxer2d_enc_losses[loss + "_enc"] = LabelLoss(
                    1, 0.1, iter_per_update
                )
            elif loss == "fed_labels":
                self.boxer2d_losses[loss] = FedLabelLoss(
                    num_classes,
                    dataset_name,
                    freq_weight_power,
                    num_fed_loss_classes,
                )
                self.boxer2d_enc_losses[loss + "_enc"] = BCELabelLoss(1)
            elif loss == "bce_labels":
                self.boxer2d_losses[loss] = BCELabelLoss(num_classes)
                self.boxer2d_enc_losses[loss + "_enc"] = BCELabelLoss(1)
            elif loss == "masks":
                focal_label = "focal_labels" in losses
                self.boxer2d_losses[loss] = PanopticMaskLoss(
                    num_points,
                    oversample_ratio,
                    importance_sample_ratio,
                    focal_label=focal_label,
                    use_uncertainty=use_uncertainty,
                )
                self.boxer2d_enc_losses[loss + "_enc"] = PanopticMaskLoss(
                    num_points,
                    oversample_ratio,
                    importance_sample_ratio,
                    focal_label=focal_label,
                    use_uncertainty=use_uncertainty,
                )
            else:
                raise ValueError(
                    "Only boxes|focal_labels|masks are supported for universal_boxer2d "
                    "losses. Found {}".format(loss)
                )

    def get_target_classes(self):
        for kk in self.boxer2d_losses.keys():
            if "labels" in kk:
                return (
                    self.boxer2d_losses[kk].src_logits,
                    self.boxer2d_losses[kk].target_classes,
                )

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs" and k != "dn_outputs"
        }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if "num_boxes" not in outputs:
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor(
                [num_boxes],
                dtype=torch.float,
                device=next(iter(outputs.values())).device,
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            num_boxes = outputs["num_boxes"].item()

        # Compute all the requested losses
        losses = {}

        if "enc_outputs" in outputs:
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])

            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                indices = self.matcher(enc_outputs, bin_targets)
                for loss in self.losses:
                    if loss == "masks" and "pred_masks" not in enc_outputs:
                        continue
                    l_dict = self.boxer2d_enc_losses[loss + "_enc"](
                        enc_outputs, bin_targets, indices, num_boxes
                    )
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.boxer2d_losses[loss](
                        aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        for loss in self.losses:
            losses.update(
                self.boxer2d_losses[loss](outputs, targets, indices, num_boxes)
            )

        return losses


@register_loss("boxer2d")
class Boxer2DLoss(BaseLoss):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses,
        instance_mask,
        iter_per_update,
    ):
        defaults = dict(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            mask_size=(instance_mask * 2),
            iter_per_update=iter_per_update,
        )
        super().__init__("boxer2d", defaults)

        self.boxer2d_losses = nn.ModuleDict()
        self.boxer2d_enc_losses = nn.ModuleDict()
        for loss in losses:
            if loss == "boxes":
                self.boxer2d_losses[loss] = BoxesLoss()
                self.boxer2d_enc_losses[loss + "_enc"] = BoxesLoss()
            elif loss == "focal_labels":
                self.boxer2d_losses[loss] = FocalLabelLoss(num_classes, 0.25)
                self.boxer2d_enc_losses[loss + "_enc"] = FocalLabelLoss(1, 0.25)
            elif loss == "labels":
                self.boxer2d_losses[loss] = LabelLoss(num_classes, 0.1, iter_per_update)
                self.boxer2d_enc_losses[loss + "_enc"] = LabelLoss(
                    1, 0.1, iter_per_update
                )
            elif loss == "masks":
                focal_label = "focal_labels" in losses
                self.boxer2d_losses[loss] = InstanceMaskLoss(
                    self.mask_size, focal_label=focal_label
                )
            else:
                raise ValueError(
                    "Only boxes|focal_labels|masks are supported for boxer2d "
                    "losses. Found {}".format(loss)
                )

    def get_target_classes(self):
        for kk in self.boxer2d_losses.keys():
            if "labels" in kk:
                return (
                    self.boxer2d_losses[kk].src_logits,
                    self.boxer2d_losses[kk].target_classes,
                )

    def prep_for_dn(self, mask_dict):
        known_indices = mask_dict["known_indices"]
        scalar, pad_size = mask_dict["scalar"], mask_dict["pad_size"]
        assert pad_size % scalar == 0
        single_pad = pad_size // scalar

        num_tgt = known_indices.numel()

        return num_tgt, single_pad, scalar

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        current_device = next(iter(outputs.values())).device

        if "dn_outputs" in outputs:
            mask_dict = outputs.pop("mask_dict")
            exc_idx = []
            if mask_dict is not None:
                num_tgt, single_pad, scalar = self.prep_for_dn(mask_dict)
            else:
                scalar = 1

            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(
                        0,
                        len(targets[i]["labels"]),
                        dtype=torch.long,
                        device=current_device,
                    )
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).to(
                        dtype=torch.long, device=current_device
                    ).unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor(
                        [], dtype=torch.long, device=current_device
                    )

                exc_idx.append((output_idx, tgt_idx))

        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs" and k != "dn_outputs"
        }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if "num_boxes" not in outputs:
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor(
                [num_boxes],
                dtype=torch.float,
                device=current_device,
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            num_boxes = outputs["num_boxes"].item()

        # Compute all the requested losses
        losses = {}

        if "enc_outputs" in outputs:
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])

            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                indices = self.matcher(enc_outputs, bin_targets)
                for loss in self.losses:
                    if loss == "masks":
                        continue

                    l_dict = self.boxer2d_enc_losses[loss + "_enc"](
                        enc_outputs, bin_targets, indices, num_boxes
                    )
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        with torch.no_grad():
            if "masks" in self.losses:
                for t in targets:
                    instance_masks = box_ops.extract_grid(
                        t["masks"][:, None].float(),
                        None,
                        t["boxes"][:, None],
                        self.mask_size,
                    )
                    t["instance_masks"] = instance_masks.squeeze(1)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.boxer2d_losses[loss](
                        aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "dn_outputs" in outputs:
            for i, dn_outputs in enumerate(outputs["dn_outputs"]):
                for loss in self.losses:
                    l_dict = self.boxer2d_losses[loss](
                        dn_outputs, targets, exc_idx, num_boxes * scalar
                    )
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        for loss in self.losses:
            losses.update(
                self.boxer2d_losses[loss](outputs, targets, indices, num_boxes)
            )

        return losses


@register_loss("boxer3d")
class Boxer3DLoss(BaseLoss):
    def __init__(self, num_classes, matcher, weight_dict, losses):
        defaults = dict(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
        )
        super().__init__("boxer3d", defaults)

        self.boxer3d_losses = nn.ModuleDict()
        self.boxer3d_enc_losses = nn.ModuleDict()
        for loss in losses:
            if loss == "boxes":
                self.boxer3d_losses[loss] = Boxes3DLoss()
                self.boxer3d_enc_losses[loss + "_enc"] = Boxes3DLoss()
            elif loss == "focal_labels":
                self.boxer3d_losses[loss] = FocalLabelLoss(num_classes, 0.25)
                self.boxer3d_enc_losses[loss + "_enc"] = FocalLabelLoss(1, 0.25)
            else:
                raise ValueError(
                    "Only boxes|focal_labels are supported for boxer3d "
                    "losses. Found {}".format(loss)
                )

    def get_target_classes(self):
        for kk in self.boxer3d_losses.keys():
            if "labels" in kk:
                return (
                    self.boxer3d_losses[kk].src_logits,
                    self.boxer3d_losses[kk].target_classes,
                )

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if "num_boxes" not in outputs:
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor(
                [num_boxes],
                dtype=torch.float,
                device=next(iter(outputs.values())).device,
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            num_boxes = outputs["num_boxes"].item()

        # Compute all the requested losses
        losses = {}

        if "enc_outputs" in outputs:
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])

            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                indices = self.matcher(enc_outputs, bin_targets)
                for loss in self.losses:
                    l_dict = self.boxer3d_enc_losses[loss + "_enc"](
                        enc_outputs, bin_targets, indices, num_boxes
                    )
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.boxer3d_losses[loss](
                        aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        for loss in self.losses:
            losses.update(
                self.boxer3d_losses[loss](outputs, targets, indices, num_boxes)
            )

        return losses
