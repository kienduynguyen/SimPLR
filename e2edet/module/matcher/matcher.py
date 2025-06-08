# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from e2edet.utils.box_ops import (
    bce_mask_cost,
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    point_sample,
    focal_mask_cost,
    dice_cost,
    focal_cls_cost,
    ce_cls_cost,
    bce_cls_cost,
)
from e2edet.utils.det3d.box_ops import box_cxcyczlwh_to_xyxyxy, generalized_box3d_iou


class BaseMatcher(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        self.params = params
        for kk, vv in params.items():
            setattr(self, kk, vv)

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for kk, vv in self.params.items():
            format_string += "{}={},".format(kk, vv)
        format_string += ")"

        return format_string


class HungarianMatcherWithMask(BaseMatcher):
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_dice: float = 1,
        cost_mask: float = 1,
        num_points: float = 0,
        loss_mode: str = "focal",
    ):
        assert (
            cost_class != 0
            or cost_bbox != 0
            or cost_giou != 0
            or cost_mask != 0
            or cost_dice != 0
        ), "all costs cant be 0"
        assert loss_mode in ("ce", "bce", "focal")

        defaults = dict(
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            cost_mask=cost_mask,
            cost_dice=cost_dice,
            num_points=num_points,
            loss_mode=loss_mode,
        )
        super().__init__(defaults)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.loss_mode == "ce":
            out_prob = outputs["pred_logits"].softmax(dim=-1)
        else:
            out_prob = outputs["pred_logits"].sigmoid()

        if "pred_boxes" in outputs:
            out_bbox = outputs["pred_boxes"]  # [batch_size * num_queries, 4]
            # [batch_size, num_target_boxes, 4]
            tgt_bbox = [v["boxes"] for v in targets]

        if "pred_masks" in outputs:
            out_mask = outputs["pred_masks"]
            out_padding_masks = outputs["padding_masks"]
            tgt_mask = [v["masks"] for v in targets]

        # Also concat the target labels and boxes
        # [batch_size, num_target_boxes]
        tgt_ids = [v["labels"] for v in targets]

        C = []
        for i in range(bs):
            if "pred_masks" in outputs:
                point_coords = torch.rand(1, self.num_points, 2, device=out_prob.device)
                tgt_mask_i = tgt_mask[i].to(out_prob)
                tgt_mask_i = tgt_mask_i[:, None]

                # get gt labels
                tgt_mask_i = point_sample(
                    tgt_mask_i,
                    point_coords.repeat(tgt_mask_i.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask_i = out_mask[i]
                out_padding_mask_i = (
                    None if out_padding_masks is None else out_padding_masks[i]
                )

                out_mask_i = out_mask_i[:, None]

                out_mask_i = point_sample(
                    out_mask_i,
                    point_coords.repeat(out_mask_i.shape[0], 1, 1),
                    input_mask=out_padding_mask_i,
                    align_corners=False,
                ).squeeze(1)

            with torch.autocast(device_type="cuda", enabled=False):
                if "pred_masks" in outputs:
                    out_mask_i = out_mask_i.float()
                    # tgt_mask_i = (tgt_mask_i >= 0.5).float()
                    tgt_mask_i = tgt_mask_i.float()
                    if self.loss_mode in ("bce", "ce"):
                        cost_mask = bce_mask_cost(out_mask_i, tgt_mask_i)
                    else:
                        cost_mask = focal_mask_cost(out_mask_i, tgt_mask_i)
                    cost_dice = dice_cost(out_mask_i, tgt_mask_i)

                out_prob_i = out_prob[i].float()

                if "pred_boxes" in outputs:
                    out_bbox_i = out_bbox[i].float()
                    tgt_bbox_i = tgt_bbox[i].float()

                    # Compute the L1 cost between boxes
                    # [batch_size * num_queries, batch_size * num_target_boxes]
                    cost_bbox = torch.cdist(out_bbox_i, tgt_bbox_i, p=1)

                    # Compute the giou cost betwen boxes
                    # [batch_size * num_queries, batch_size * num_target_boxes]
                    cost_giou = -generalized_box_iou(
                        box_cxcywh_to_xyxy(out_bbox_i), box_cxcywh_to_xyxy(tgt_bbox_i)
                    )

                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                # [batch_size * num_queries, batch_size * num_target_boxes]
                if self.loss_mode == "ce":
                    cost_class = ce_cls_cost(out_prob_i, tgt_ids[i])
                elif self.loss_mode == "bce":
                    cost_class = bce_cls_cost(out_prob_i, tgt_ids[i])
                else:
                    cost_class = focal_cls_cost(out_prob_i, tgt_ids[i])

                # Final cost matrix
                C_i = self.cost_class * cost_class
                if "pred_boxes" in outputs:
                    C_i += self.cost_giou * cost_giou + self.cost_bbox * cost_bbox

                if "pred_masks" in outputs or "feat_masks" in outputs:
                    C_i += self.cost_mask * cost_mask + self.cost_dice * cost_dice
                # [num_queries, num_target_boxes]
                C_i = C_i.view(num_queries, -1).cpu()
                C.append(C_i)

        indices = [linear_sum_assignment(c) for c in C]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class HungarianMatcherWithMaskV2(BaseMatcher):
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_dice: float = 1,
        cost_mask: float = 1,
        num_points: float = 0,
        loss_mode: str = "focal",
    ):
        assert (
            cost_class != 0
            or cost_bbox != 0
            or cost_giou != 0
            or cost_mask != 0
            or cost_dice != 0
        ), "all costs cant be 0"
        assert loss_mode in ("ce", "bce", "focal")

        defaults = dict(
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            cost_mask=cost_mask,
            cost_dice=cost_dice,
            num_points=num_points,
            loss_mode=loss_mode,
        )
        super().__init__(defaults)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.loss_mode == "ce":
            out_prob = outputs["pred_logits"].softmax(dim=-1)
        else:
            out_prob = outputs["pred_logits"].sigmoid()

        if "pred_boxes" in outputs:
            out_bbox = outputs["pred_boxes"]  # [batch_size * num_queries, 4]
            # [batch_size, num_target_boxes, 4]
            tgt_bbox = [v["boxes"] for v in targets]

        out_mask = outputs["pred_masks"]
        out_padding_masks = outputs["padding_masks"]
        tgt_mask = [v["masks"] for v in targets]

        # Also concat the target labels and boxes
        # [batch_size, num_target_boxes]
        tgt_ids = [v["labels"] for v in targets]

        C = []
        for i in range(bs):
            point_coords = torch.rand(1, self.num_points, 2, device=out_prob.device)
            tgt_mask_i = tgt_mask[i].to(out_prob)
            tgt_mask_i = tgt_mask_i[:, None]

            # get gt labels
            tgt_mask_i = point_sample(
                tgt_mask_i,
                point_coords.repeat(tgt_mask_i.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask_i = out_mask[i]
            out_padding_mask_i = (
                None if out_padding_masks is None else out_padding_masks[i]
            )

            out_mask_i = out_mask_i[:, None]

            out_mask_i = point_sample(
                out_mask_i,
                point_coords.repeat(out_mask_i.shape[0], 1, 1),
                input_mask=out_padding_mask_i,
                align_corners=False,
            ).squeeze(1)

            with torch.autocast(device_type="cuda", enabled=False):
                out_mask_i = out_mask_i.float()
                # tgt_mask_i = (tgt_mask_i >= 0.5).float()
                tgt_mask_i = tgt_mask_i.float()
                cost_mask = bce_mask_cost(out_mask_i, tgt_mask_i)
                cost_dice = dice_cost(out_mask_i, tgt_mask_i)

                out_prob_i = out_prob[i].float()

                if "pred_boxes" in outputs:
                    out_bbox_i = out_bbox[i].float()
                    tgt_bbox_i = tgt_bbox[i].float()

                    # Compute the L1 cost between boxes
                    # [batch_size * num_queries, batch_size * num_target_boxes]
                    cost_bbox = torch.cdist(out_bbox_i, tgt_bbox_i, p=1)

                    # Compute the giou cost betwen boxes
                    # [batch_size * num_queries, batch_size * num_target_boxes]
                    cost_giou = -generalized_box_iou(
                        box_cxcywh_to_xyxy(out_bbox_i), box_cxcywh_to_xyxy(tgt_bbox_i)
                    )

                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                # [batch_size * num_queries, batch_size * num_target_boxes]
                if self.loss_mode == "ce":
                    cost_class = ce_cls_cost(out_prob_i, tgt_ids[i])
                elif self.loss_mode == "bce":
                    cost_class = bce_cls_cost(out_prob_i, tgt_ids[i])
                else:
                    cost_class = focal_cls_cost(out_prob_i, tgt_ids[i])

                # Final cost matrix
                C_i = self.cost_class * cost_class
                if "pred_boxes" in outputs:
                    C_i += self.cost_giou * cost_giou + self.cost_bbox * cost_bbox

                if "pred_masks" in outputs or "feat_masks" in outputs:
                    C_i += self.cost_mask * cost_mask + self.cost_dice * cost_dice
                # [num_queries, num_target_boxes]
                C_i = C_i.view(num_queries, -1).cpu()
                C.append(C_i)

        indices = [linear_sum_assignment(c) for c in C]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class HungarianMatcher(BaseMatcher):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        loss_mode: str = "focal",
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"
        assert loss_mode in ("ce", "bce", "focal")

        defaults = dict(
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            loss_mode=loss_mode,
        )
        super().__init__(defaults)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.loss_mode == "ce":
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(dim=-1)
            )  # [batch_size * num_queries, num_classes]
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        # [batch_size * num_target_boxes]
        tgt_ids = torch.cat([v["labels"] for v in targets])
        # [batch_size * num_target_boxes, 4]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        with torch.autocast(device_type="cuda", enabled=False):
            out_prob = out_prob.float()
            out_bbox = out_bbox.float()
            tgt_bbox = tgt_bbox.float()

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # [batch_size * num_queries, batch_size * num_target_boxes]
            if self.loss_mode == "ce":
                cost_class = ce_cls_cost(out_prob, tgt_ids)
            elif self.loss_mode == "bce":
                cost_class = bce_cls_cost(out_prob, tgt_ids)
            else:
                cost_class = focal_cls_cost(out_prob, tgt_ids)

            # Compute the L1 cost between boxes
            # [batch_size * num_queries, batch_size * num_target_boxes]
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            # [batch_size * num_queries, batch_size * num_target_boxes]
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
            )

            # Final cost matrix
            C = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )

            # [batch_size, num_queries, batch_size * num_target_boxes]
            C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class HungarianMatcher3d(BaseMatcher):
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_rad: float = 1,
    ):
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_rad != 0
        ), "all costs cant be 0"
        defaults = dict(
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            cost_rad=cost_rad,
        )
        super().__init__(defaults)

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].sigmoid()
        # ([batch_size, num_queries, 6], [batch_size, num_queries, 2])
        out_bbox, out_rad = outputs["pred_boxes"].split(6, dim=-1)

        # Also concat the target labels and boxes
        # [batch_size, num_target_boxes]
        tgt_ids = [v["labels"] for v in targets]
        # [batch_size, num_target_boxes, 6]
        tgt_bbox = [v["boxes"][..., :6] for v in targets]
        # [batch_size, num_target_boxes, 2]
        tgt_rad = [v["boxes"][..., 6:] for v in targets]

        alpha = 0.25
        gamma = 2.0

        C = []
        for i in range(bs):
            with torch.autocast(device_type="cuda", enabled=False):
                out_bbox_i = out_bbox[i].float()
                out_rad_i = out_rad[i].float()
                out_prob_i = out_prob[i].float()
                tgt_bbox_i = tgt_bbox[i].float()
                tgt_rad_i = tgt_rad[i].float()

                # [num_queries, num_target_boxes]
                cost_giou = -generalized_box3d_iou(
                    box_cxcyczlwh_to_xyxyxy(out_bbox_i),
                    box_cxcyczlwh_to_xyxyxy(tgt_bbox_i),
                )

                neg_cost_class = (
                    (1 - alpha) * (out_prob_i**gamma) * (-(1 - out_prob_i + 1e-8).log())
                )
                pos_cost_class = (
                    alpha * ((1 - out_prob_i) ** gamma) * (-(out_prob_i + 1e-8).log())
                )
                cost_class = (
                    pos_cost_class[:, tgt_ids[i]] - neg_cost_class[:, tgt_ids[i]]
                )

                # Compute the L1 cost between boxes
                # [num_queries, num_target_boxes]
                cost_bbox = torch.cdist(out_bbox_i, tgt_bbox_i, p=1)
                cost_rad = torch.cdist(out_rad_i, tgt_rad_i, p=1)

            # Final cost matrix
            C_i = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
                + self.cost_rad * cost_rad
            )
            # [num_queries, num_target_boxes]
            C_i = C_i.view(num_queries, -1).cpu()
            C.append(C_i)

        indices = [linear_sum_assignment(c) for c in C]

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def build_matcher(config):
    matcher_type = config.type
    params = config.params
    # matcher_type, class_weight, bbox_weight, giou_weight, use_bcl
    if matcher_type == "hungarian3d":
        matcher_cls = HungarianMatcher3d
    elif matcher_type == "hungarian":
        matcher_cls = HungarianMatcher
    elif matcher_type == "hungarian_w_masks":
        matcher_cls = HungarianMatcherWithMask
    elif matcher_type == "hungarian_w_masks_v2":
        matcher_cls = HungarianMatcherWithMaskV2
    else:
        raise ValueError(
            f"Only hungarian3d|hungarian|hungarian_w_masks accepted, got {matcher_type}"
        )
    matcher = matcher_cls(**params)

    return matcher
