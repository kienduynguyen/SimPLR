import torch
import torch.nn as nn
import torch.nn.functional as F

from e2edet.utils import box_ops
from e2edet.utils.det3d import box_ops as box3d_ops
from e2edet.utils.general import concat_and_pad_masks
from e2edet.dataset.helper.categories import LVIS_CATEGORY_IMAGE_COUNT


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )  # [batch_size * num_target_boxes]
    src_idx = torch.cat(
        [src for (src, _) in indices]
    )  # [batch_size * num_target_boxes]
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat(
        [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
    )  # [batch_size * num_target_boxes]
    tgt_idx = torch.cat(
        [tgt for (_, tgt) in indices]
    )  # [batch_size * num_target_boxes]
    return batch_idx, tgt_idx


def get_fed_loss_cls_weights(class_image_count, freq_weight_power=1.0):
    """
    Get frequency weight for each class sorted by class id.
    We now calcualte freqency weight using image_count to the power freq_weight_power.
    Args:
        class_image_count: number of images containing each class
        freq_weight_power: power value
    """
    max_id = max(c["id"] for c in class_image_count)
    class_freq = torch.zeros(max_id + 1)

    for c in class_image_count:
        class_freq[c["id"]] = c["image_count"]
    class_freq_weight = class_freq.float() ** freq_weight_power

    return class_freq_weight


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    inputs = inputs.float()
    targets = targets.float()

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t).pow(gamma))

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum() / num_boxes


sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.sum() / num_masks


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_boxes: float):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.float()
    targets = targets.float()

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


class BaseLoss(nn.Module):
    def __init__(self, name, params=None):
        super().__init__()
        self.name = name
        self.params = params
        for kk, vv in params.items():
            setattr(self, kk, vv)

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for kk, vv in self.params.items():
            format_string += "{}={},".format(kk, vv)
        format_string += ")"

        return format_string


class LabelLoss(BaseLoss):
    def __init__(self, num_classes, eos_coef, iter_per_update):
        defaults = dict(
            num_classes=num_classes, eos_coef=eos_coef, iter_per_update=iter_per_update
        )
        super().__init__("label_loss", defaults)

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.target_classes = None
        self.src_logits = None

    def loss_fields(self):
        return ("loss_ce",)

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        self.target_classes = target_classes_o
        self.src_logits = src_logits[idx]

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )  # batch_size x num_queries

        # assign correct classes to matched queries
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
        )
        losses = {"loss_ce": (loss_ce / self.iter_per_update)}

        return losses


class FocalLabelLoss(BaseLoss):
    def __init__(self, num_classes, focal_alpha):
        defaults = dict(
            num_classes=num_classes,
            focal_alpha=focal_alpha,
        )
        super().__init__("focal_label_loss", defaults)

        self.target_classes = None
        self.src_logits = None

    def loss_fields(self):
        return ("loss_ce",)

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        self.target_classes = target_classes_o
        self.src_logits = src_logits[idx]
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=2,
        )
        losses = {"loss_ce": loss_ce}

        return losses


class BCELabelLoss(BaseLoss):
    def __init__(self, num_classes):
        defaults = dict(num_classes=num_classes)
        super().__init__("bce_label_loss", defaults)

        self.target_classes = None
        self.src_logits = None

    def loss_fields(self):
        return ("loss_ce",)

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        self.target_classes = target_classes_o
        self.src_logits = src_logits[idx]
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_ce_loss(src_logits, target_classes_onehot, num_boxes)
        losses = {"loss_ce": loss_ce}

        return losses


class FedLabelLoss(BaseLoss):
    def __init__(
        self,
        num_classes,
        dataset_name,
        freq_weight_power=1.0,
        num_fed_loss_classes=50,
    ):
        if dataset_name == "lvis":
            class_image_count = LVIS_CATEGORY_IMAGE_COUNT
        else:
            raise ValueError("Only lvis is supported")
        fed_loss_cls_weights = get_fed_loss_cls_weights(
            class_image_count, freq_weight_power
        )
        defaults = dict(
            num_classes=num_classes,
            num_fed_loss_classes=num_fed_loss_classes,
        )

        super().__init__("fed_label_loss", defaults)
        self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)

        self.target_classes = None
        self.src_logits = None

    def loss_fields(self):
        return ("loss_ce",)

    def get_fed_loss_classes(
        self, gt_classes, num_fed_loss_classes, num_classes, weight
    ):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes
        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes

        return fed_loss_classes

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        self.target_classes = target_classes_o
        self.src_logits = src_logits[idx]
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = F.binary_cross_entropy_with_logits(
            src_logits, target_classes_onehot, reduction="none"
        )

        fed_loss_classes = self.get_fed_loss_classes(
            target_classes_o,
            self.num_fed_loss_classes,
            self.num_classes,
            self.fed_loss_cls_weights,
        )
        fed_loss_classes_mask = fed_loss_classes.new_zeros(self.num_classes + 1)
        fed_loss_classes_mask[fed_loss_classes] = 1
        fed_loss_classes_mask = fed_loss_classes_mask[: self.num_classes]
        weight = fed_loss_classes_mask.view(1, 1, self.num_classes).float()

        loss_ce = torch.sum(loss_ce * weight) / (
            src_logits.shape[0] * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses


class BoxesLoss(BaseLoss):
    def __init__(self):
        defaults = dict()
        super().__init__("boxes_loss", defaults)

    def loss_fields(self):
        return ("loss_bbox", "loss_giou")

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = _get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]  # batch_size * nb_target_boxes x 4
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )  # batch_size * nb_target_boxes x 4

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses


class Boxes3DLoss(BaseLoss):
    def __init__(self):
        defaults = dict()
        super().__init__("boxes3d_loss", defaults)

    def loss_fields(self):
        return ("loss_bbox", "loss_giou", "loss_rad")

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = _get_src_permutation_idx(indices)
        src_boxes, src_rads = outputs["pred_boxes"][idx].split(6, dim=-1)
        target_boxes = torch.cat(
            [t["boxes"][i][..., :6] for t, (_, i) in zip(targets, indices)], dim=0
        )  # batch_size * nb_target_boxes x 6

        target_rads = torch.cat(
            [t["boxes"][i][..., 6:] for t, (_, i) in zip(targets, indices)], dim=0
        )  # batch_size * nb_target_boxes x (1, 2)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_rad = F.l1_loss(src_rads, target_rads, reduction="none")

        losses = {}

        loss_giou = 1 - torch.diag(
            box3d_ops.generalized_box3d_iou(
                box3d_ops.box_cxcyczlwh_to_xyxyxy(src_boxes),
                box3d_ops.box_cxcyczlwh_to_xyxyxy(target_boxes),
            )
        )
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        losses["loss_rad"] = loss_rad.sum() / num_boxes

        return losses


class SemanticMaskLoss(BaseLoss):
    def __init__(self, focal_label=True):
        defaults = dict(focal_label=focal_label)
        super().__init__("semantic_mask_loss", defaults)

    def loss_fields(self):
        return ("loss_mask", "loss_dice")

    def forward(self, outputs, targets, indices, num_boxes):
        assert "feat_masks" in outputs

        output, feat = outputs["feat_masks"]
        src_masks = torch.cat(
            [
                torch.einsum("qc,chw->qhw", output[i][src], feat[i])
                for i, (src, _) in enumerate(indices)
            ]
        ) + (
            output[0][0].sum() * 0
        )  # hack for gradient

        target_masks, _ = concat_and_pad_masks(
            [t["masks"][i] for t, (_, i) in zip(targets, indices)]
        )

        target_masks = F.interpolate(
            target_masks[:, None].to(src_masks),
            size=(src_masks.shape[1], src_masks.shape[2]),
            mode="nearest",
        )

        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        if not self.focal_label:
            losses = {
                "loss_mask": sigmoid_ce_loss(src_masks, target_masks, num_boxes)
                / src_masks.shape[1],
                "loss_dice": dice_loss_jit(src_masks, target_masks, num_boxes),
            }
        else:
            losses = {
                "loss_mask": sigmoid_focal_loss_jit(src_masks, target_masks, num_boxes)
                / src_masks.shape[1],
                "loss_dice": dice_loss_jit(src_masks, target_masks, num_boxes),
            }

        return losses


class InstanceMaskLoss(BaseLoss):
    def __init__(self, mask_size, focal_label=True):
        defaults = dict(mask_size=mask_size, focal_label=focal_label)
        super().__init__("instance_mask_loss", defaults)

    def loss_fields(self):
        return ("loss_mask", "loss_dice")

    def forward(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        idx = _get_src_permutation_idx(indices)

        target_masks = torch.cat(
            [t["instance_masks"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        src_masks = outputs["pred_masks"]
        src_masks = src_masks[idx].flatten(1)
        target_masks = target_masks.flatten(1)

        if not self.focal_label:
            losses = {
                "loss_mask": sigmoid_ce_loss(src_masks, target_masks, num_boxes)
                / (self.mask_size**2),
                "loss_dice": dice_loss_jit(src_masks, target_masks, num_boxes),
            }
        else:
            losses = {
                "loss_mask": sigmoid_focal_loss_jit(src_masks, target_masks, num_boxes)
                / (self.mask_size**2),
                "loss_dice": dice_loss_jit(src_masks, target_masks, num_boxes),
            }

        return losses


class PanopticMaskLoss(BaseLoss):
    def __init__(
        self,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        focal_label=True,
        use_uncertainty=True,
    ):
        defaults = dict(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            focal_label=focal_label,
            use_uncertainty=use_uncertainty,
        )
        super().__init__("panoptic_mask_loss", defaults)

    def loss_fields(self):
        return ("loss_mask", "loss_dice")

    def forward(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        idx = _get_src_permutation_idx(indices)

        src_masks = outputs["pred_masks"][idx]
        src_padding_masks = outputs["padding_masks"]

        if outputs["padding_masks"] is not None:
            src_padding_masks = src_padding_masks[idx]
        else:
            src_padding_masks = None

        with torch.no_grad():
            target_masks, target_padding_masks = concat_and_pad_masks(
                [t["masks"][i] for t, (_, i) in zip(targets, indices)]
            )
            if src_padding_masks is None:
                target_padding_masks = None

            if self.use_uncertainty:
                point_coords = box_ops.get_uncertain_point_coords_with_randomness(
                    src_masks[:, None],
                    lambda logits: box_ops.calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                    logits_mask=src_padding_masks,
                )
            else:
                point_coords = box_ops.get_error_point_coords_with_randomness(
                    src_masks[:, None],
                    box_ops.calculate_error,
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                    target_masks[:, None].to(src_masks),
                    logits_mask=src_padding_masks,
                    targets_mask=target_padding_masks,
                )

            target_masks = box_ops.point_sample(
                target_masks[:, None].to(src_masks),
                point_coords,
                input_mask=target_padding_masks,
                align_corners=False,
            )
            # target_masks = (target_masks >= 0.5).float()

        src_masks = box_ops.point_sample(
            src_masks[:, None],
            point_coords,
            input_mask=src_padding_masks,
            align_corners=False,
        )

        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        if not self.focal_label:
            losses = {
                "loss_mask": sigmoid_ce_loss(src_masks, target_masks, num_boxes)
                / src_masks.shape[1],
                "loss_dice": dice_loss_jit(src_masks, target_masks, num_boxes),
            }
        else:
            losses = {
                "loss_mask": sigmoid_focal_loss_jit(src_masks, target_masks, num_boxes)
                / src_masks.shape[1],
                "loss_dice": dice_loss_jit(src_masks, target_masks, num_boxes),
            }

        return losses
