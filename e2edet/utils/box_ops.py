from typing import List

import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_area


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def box_xywh_to_xyxy(x) -> torch.Tensor:
    x, y, w, h = x.unbind(-1)

    return torch.stack((x, y, x + w, y + h), dim=-1)


def box_cxcywh_to_xyxy(x) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_intersect(boxes1, boxes2) -> torch.Tensor:
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter


def sample_boxes(num_sample, num_boxes, device, dtype):
    boxes = torch.rand(num_sample, num_boxes, 4, device=device, dtype=dtype)
    boxes = box_cxcywh_to_xyxy(boxes).clamp(min=0.01, max=0.99)

    return box_xyxy_to_cxcywh(boxes)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2) -> List[torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    inter = box_intersect(boxes1, boxes2)

    union = area1[:, None] + area2 - inter
    iou = inter / union

    return iou, union


def box_iou_detectron(boxes1, boxes2) -> torch.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    inter = box_intersect(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def generalized_box_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def dice_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    inputs: [N, C]
    targets: [M, C]
    """
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)

    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1).unsqueeze(1) + targets.sum(-1).unsqueeze(0)

    return -(numerator + 1) / (denominator + 1)


@torch.jit.script
def focal_mask_cost(
    inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """
    inputs: [N, C]
    targets: [M, C]
    """
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    hw = inputs.shape[1]

    neg_cost = (1 - alpha) * (inputs**gamma) * (-(1 - inputs + 1e-8).log())
    pos_cost = alpha * ((1 - inputs) ** gamma) * (-(inputs + 1e-8).log())

    cost = torch.einsum("nc,mc->nm", pos_cost, targets) + torch.einsum(
        "nc,mc->nm", neg_cost, (1 - targets)
    )

    return cost / hw


def focal_mask_cost_v2(
    inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """
    inputs: [N, C]
    targets: [M, C]
    """
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    hw = inputs.shape[1]

    neg_cost = (1 - alpha) * (inputs**gamma) * (-(1 - inputs + 1e-8).log())
    pos_cost = alpha * ((1 - inputs) ** gamma) * (-(inputs + 1e-8).log())

    cost = torch.einsum("nc,mc->nm", pos_cost, 2 * (targets - 0.5)) + torch.einsum(
        "nc,mc->nm", neg_cost, 2 * (0.5 - targets)
    )

    return cost / hw


def bce_mask_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


@torch.jit.script
def focal_cls_cost(
    out_prob: torch.Tensor,
    tgt_ids: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    out_prob: [N, C]
    tgt_ids: [N,]
    """
    neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
    cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

    return cost_class


def bce_cls_cost(out_prob: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
    neg_cost_class = -(1 - out_prob + 1e-8).log()
    pos_cost_class = -(out_prob + 1e-8).log()
    cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

    return cost_class


def ce_cls_cost(out_prob: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
    """
    out_prob: [N, C]
    tgt_ids: [N,]
    """
    cost_class = -out_prob[:, tgt_ids]

    return cost_class


def masks_to_boxes(masks) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks,
        (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros(0, 4, device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(h, dtype=torch.float)
    x = torch.arange(w, dtype=torch.float)
    y, x = torch.meshgrid(y, x, indexing="ij")

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def proposals_to_boxes(proposals, masks) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks
    The proposals should be in format [N, L, H, W] where N is the number of masks,
        (H, W) are the spatial dimensions.
    The masks should be in format [N, H, W]
    Returns a [N, L, 4] tensors, with the boxes in xyxy format
    """
    if proposals.numel() == 0:
        return torch.zeros(0, 0, 4, device=proposals.device)

    n, l, h, w = proposals.shape

    if masks is None:
        not_masks = proposals.new_ones(n, h, w, dtype=torch.bool)
    else:
        not_masks = ~masks

    y = torch.arange(h, dtype=proposals.dtype, device=proposals.device)
    x = torch.arange(w, dtype=proposals.dtype, device=proposals.device)
    y, x = torch.meshgrid(y, x, indexing="ij")

    proposals = proposals * not_masks.float().unsqueeze(1)
    proposals = (proposals.view(n * l, h, w) >= 0.5).float()

    x_mask = proposals * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(proposals.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = proposals * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(proposals.bool()), 1e8).flatten(1).min(-1)[0]

    boxes = torch.stack([x_min, y_min, x_max, y_max], 1).view(n, l, 4)
    img_h = not_masks[:, :, 0].sum(dim=-1, dtype=proposals.dtype)
    img_w = not_masks[:, 0, :].sum(dim=-1, dtype=proposals.dtype)
    img_size = torch.stack([img_w, img_h, img_w, img_h], dim=-1).unsqueeze(1)

    return boxes / img_size


def iou_with_ign(boxes1, boxes2) -> torch.Tensor:
    """
    Computes the amount of overlap of boxes2 has within boxes1, which is handy for dealing with
    ignore areas. Hence, assume that boxes2 are ignore regions and boxes1 are anchor boxes, then
    we may want to know how much overlap the anchors have inside the ignore regions boxes2.
    boxes1: (M, 4) [x1, y1, x2, y2]
    boxes2: (N, 4) [x1, y1, x2, y2]
    mode: if 'elementwise', M needs to be equal to N and we compute
        intersection of M pairs in boxes1 and boxes2 elementwise. Otherwise,
        we compute intersection of NxM pairs.
    """
    area1 = box_area(boxes1)
    intersect = box_intersect(boxes1, boxes2)
    iou_w_ign = intersect / area1

    return iou_w_ign


def point_sample_w_boxes(
    input, boxes, point_coords_in_boxes, point_random, input_mask=None, **kwargs
):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        boxes (Tensor): A tensor of shape (N, 4) for box region of mask prediction.
        point_coords_in_boxes (Tensor): A tensor of shape (N, P1, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates within box coordinates.
        point_random (Tensor): A tensor of shape (N, P2, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
        input_mask (Tensor): A mask of input (N, H, W).
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    boxes = box_cxcywh_to_xyxy(boxes)
    boxes_x1y1, boxes_x2y2 = boxes.unsqueeze(1).split(2, dim=-1)
    point_coords_in_boxes = (
        point_coords_in_boxes * (boxes_x2y2 - boxes_x1y1) + boxes_x1y1
    )
    if input_mask is not None:
        not_mask = ~input_mask
        size_h = not_mask[:, :, 0].sum(dim=1, dtype=input.dtype)
        size_w = not_mask[:, 0, :].sum(dim=1, dtype=input.dtype)
        h, w = input.shape[-2:]

        ratio_h = size_h / h
        ratio_w = size_w / w
        ratio = torch.stack([ratio_w, ratio_h], dim=-1)

        point_coords_in_boxes = point_coords_in_boxes * ratio.unsqueeze(1)
        point_random = point_random * ratio.unsqueeze(1)
    point_coords = torch.cat([point_coords_in_boxes, point_random], dim=1).unsqueeze(2)

    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs).squeeze(3)

    return output


def point_sample(input, point_coords, input_mask=None, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
        input_mask (Tensor): A mask of input (N, H, W).
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    if input_mask is not None:
        not_mask = ~input_mask
        size_h = not_mask[:, :, 0].sum(dim=1, dtype=input.dtype)
        size_w = not_mask[:, 0, :].sum(dim=1, dtype=input.dtype)
        h, w = input.shape[-2:]

        ratio_h = size_h / h
        ratio_w = size_w / w
        ratio = torch.stack([ratio_w, ratio_h], dim=-1)

        if point_coords.dim() == 3:
            point_coords = point_coords * ratio.unsqueeze(1)
        elif point_coords.dim() == 4:
            point_coords = point_coords * ratio.unsqueeze(1).unsqueeze(2)

    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)

    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_box_point_coords_with_randomness(
    coarse_logits,
    coarse_boxes,
    uncertainty_func,
    num_points,
    oversample_ratio,
    importance_sample_ratio,
    logits_mask=None,
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        coarse_boxes (Tensor): A tensor of shape (N, 4) for box region of mask prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled inside box region.
        logits_mask (Tensor): A tensor of shape (N, Hmask, Wmask) for masking the padded region.
    Returns:
        point_coords_in_boxes (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    coarse_boxes = box_cxcywh_to_xyxy(coarse_boxes)
    boxes_x1y1, boxes_x2y2 = coarse_boxes.unsqueeze(1).split(2, dim=-1)
    point_coords_in_boxes = point_coords * (boxes_x2y2 - boxes_x1y1) + boxes_x1y1

    point_logits = point_sample(
        coarse_logits,
        point_coords_in_boxes,
        input_mask=logits_mask,
        align_corners=False,
    )
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        num_boxes, dtype=torch.long, device=coarse_logits.device
    )
    idx += shift[:, None]
    point_coords_in_boxes = point_coords_in_boxes.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    point_random = None
    if num_random_points > 0:
        point_random = torch.rand(
            num_boxes, num_random_points, 2, device=coarse_logits.device
        )
        point_random_in_boxes = point_random * (boxes_x2y2 - boxes_x1y1) + boxes_x1y1
        point_coords_in_boxes = torch.cat(
            [point_coords_in_boxes, point_random_in_boxes], dim=1
        )

    return point_coords_in_boxes


def get_uncertain_point_coords_with_randomness(
    coarse_logits,
    uncertainty_func,
    num_points,
    oversample_ratio,
    importance_sample_ratio,
    logits_mask=None,
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
        logits_mask (Tensor): A tensor of shape (N, Hmask, Wmask) for masking the padded region.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(
        coarse_logits, point_coords, input_mask=logits_mask, align_corners=False
    )
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        num_boxes, dtype=torch.long, device=coarse_logits.device
    )
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(
                    num_boxes, num_random_points, 2, device=coarse_logits.device
                ),
            ],
            dim=1,
        )
    return point_coords


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def get_error_point_coords_with_randomness(
    coarse_logits,
    uncertainty_func,
    num_points,
    oversample_ratio,
    importance_sample_ratio,
    targets,
    logits_mask=None,
    targets_mask=None,
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
        logits_mask (Tensor): A tensor of shape (N, Hmask, Wmask) for masking the padded region.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(
        coarse_logits, point_coords, input_mask=logits_mask, align_corners=False
    )
    target_logits = point_sample(
        targets, point_coords, input_mask=targets_mask, align_corners=False
    )
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits, target_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        num_boxes, dtype=torch.long, device=coarse_logits.device
    )
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(
                    num_boxes, num_random_points, 2, device=coarse_logits.device
                ),
            ],
            dim=1,
        )
    return point_coords


def calculate_error(logits, targets):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone().sigmoid()

    return torch.abs(gt_class_logits - targets)


def extract_pixels(x, x_mask, boxes, pixel_box_coords, pixel_random_coords):
    """
    Params:
    :x: (B, C, H, W)
    :x_mask: (B, H, W)
    :boxes: (B, L, 4)
    :pixel_box_coords: (B, L, num_box_pixels, 2)
    :pixel_random_coords: (B, L, num_random_pixels, 2)
    Return:
    :pixel: (B, L, num_pixels, C)
    """
    b, l = boxes.shape[:2]
    c = x.shape[1]

    if isinstance(pixel_box_coords, torch.Tensor):
        assert (
            isinstance(pixel_random_coords, torch.Tensor) or pixel_random_coords is None
        )
        if b == 0:
            if pixel_random_coords is None:
                num_pixels = pixel_box_coords.shape[2]
            else:
                num_pixels = pixel_box_coords.shape[2] + pixel_random_coords.shape[2]
            return (
                torch.zeros(0, l, num_pixels, c, device=x.device, dtype=x.dtype),
                pixel_box_coords,
                pixel_random_coords,
            )

    elif isinstance(pixel_box_coords, int):
        assert isinstance(pixel_random_coords, int) or pixel_random_coords is None
        if b == 0:
            num_pixels = pixel_box_coords + pixel_random_coords
            return (
                torch.zeros(0, l, num_pixels, c, device=x.device, dtype=x.dtype),
                torch.zeros(0, l, pixel_box_coords, c, device=x.device, dtype=x.dtype),
                torch.zeros(
                    0, l, pixel_random_coords, c, device=x.device, dtype=x.dtype
                ),
            )

        pixel_box_coords = torch.rand(
            b, l, pixel_box_coords, 2, dtype=boxes.dtype, device=boxes.device
        )

        if pixel_random_coords > 0:
            pixel_random_coords = torch.rand(
                b, l, pixel_random_coords, 2, dtype=boxes.dtype, device=boxes.device
            )
        else:
            pixel_random_coords = None
    else:
        raise ValueError

    boxes = box_cxcywh_to_xyxy(boxes)
    boxes1, boxes2 = boxes.unsqueeze(-2).split(2, dim=-1)
    pixel_coords = pixel_box_coords * (boxes2 - boxes1) + boxes1
    if pixel_random_coords is not None:
        pixel_coords = torch.cat([pixel_coords, pixel_random_coords], dim=2)

    if x_mask is not None:
        not_x_mask = ~x_mask
        size_h = not_x_mask[:, :, 0].sum(dim=1, dtype=x.dtype)
        size_w = not_x_mask[:, 0, :].sum(dim=1, dtype=x.dtype)
        h, w = x.shape[-2:]

        ratio_h = size_h / h
        ratio_w = size_w / w
        ratio = torch.stack([ratio_w, ratio_h], dim=-1)

        pixel_coords = pixel_coords * ratio.unsqueeze(1).unsqueeze(2)

    pixel_coords = pixel_coords * 2 - 1
    pixels = F.grid_sample(x, pixel_coords, align_corners=False)

    return pixels.permute(0, 2, 3, 1), pixel_box_coords, pixel_random_coords


def extract_grid(
    x, x_mask, boxes, grid_size=15, align_corners=False, roi_align=False, roi_ratio=2
):
    """
    Params:
    :x: (B, C, H, W)
    :x_mask: (B, H, W)
    :boxes: (B, L, 4)
    Return:
    :grid: (B, L, grid_size, grid_size, C)
    """
    b, l = boxes.shape[:2]
    c = x.shape[1]
    if b == 0:
        return torch.zeros(
            0, l, grid_size, grid_size, c, device=x.device, dtype=x.dtype
        )

    grid_size = grid_size * roi_ratio if roi_align else grid_size

    if align_corners:
        indices = torch.arange(0, grid_size, device=x.device, dtype=x.dtype)
        step = 1.0 / (grid_size - 1)
    else:
        indices = 0.5 + torch.arange(0, grid_size, device=x.device, dtype=x.dtype)
        step = 1.0 / grid_size
    i, j = torch.meshgrid(indices, indices, indexing="ij")
    grid_indices = torch.stack([j, i], dim=-1)  # 7 x 7 x 2

    boxes = box_cxcywh_to_xyxy(boxes)
    if x_mask is not None:
        not_x_mask = ~x_mask
        size_h = not_x_mask[:, :, 0].sum(dim=1, dtype=x.dtype)
        size_w = not_x_mask[:, 0, :].sum(dim=1, dtype=x.dtype)
        h, w = x.shape[-2:]

        ratio_h = size_h / h
        ratio_w = size_w / w
        ratio = torch.stack([ratio_w, ratio_h, ratio_w, ratio_h], dim=-1)

        boxes = boxes * ratio.unsqueeze(1)

    boxes1, boxes2 = boxes.unsqueeze(-2).unsqueeze(-2).split(2, dim=-1)

    grid = grid_indices * step * (boxes2 - boxes1) + boxes1
    grid = grid * 2 - 1
    grid = grid.view(b, l, grid_size * grid_size, 2)

    grid = F.grid_sample(x, grid, align_corners=False)

    if roi_align:
        grid = grid.view(
            b,
            -1,
            l,
            grid_size // roi_ratio,
            roi_ratio,
            grid_size // roi_ratio,
            roi_ratio,
        )
        grid = grid.mean(-1).mean(-2)
    else:
        grid = grid.view(b, -1, l, grid_size, grid_size)

    return grid.permute(0, 2, 3, 4, 1)


def paste_grid(seg_mask, boxes, x_size):
    # seg_mask: l x 7 x 7
    # boxes: l x 4
    assert seg_mask.dim() == 3
    assert boxes.shape[0] == seg_mask.shape[0]
    nq = boxes.shape[0]

    h, w = x_size
    x1, y1, x2, y2 = boxes.unsqueeze(-2).unsqueeze(-2).unbind(-1)

    img_x = torch.arange(w, device=boxes.device, dtype=boxes.dtype) + 0.5
    img_y = torch.arange(h, device=boxes.device, dtype=boxes.dtype) + 0.5
    img_y, img_x = torch.meshgrid(img_y, img_x, indexing="ij")

    # l x h x w
    img_y = (img_y - y1) / (y2 - y1) * 2 - 1
    img_x = (img_x - x1) / (x2 - x1) * 2 - 1
    img_grid = torch.stack([img_x, img_y], dim=-1)
    img_grid = img_grid.view(nq, h, w, 2)

    img = F.grid_sample(seg_mask[:, None], img_grid, align_corners=False)
    img = img.view(nq, h, w)

    return img


def mask_process(pred_mask, img_size, output_height, output_width):
    pred_mask = pred_mask[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    pred_mask = F.interpolate(
        pred_mask,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )[0]

    return pred_mask


def mask_process_w_boxes(pred_mask, pred_box, img_size, output_height, output_width):
    pred_mask = pred_mask[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    pred_mask = F.interpolate(
        pred_mask,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )[0]

    y, x = torch.meshgrid(
        torch.arange(output_height, device=pred_box.device, dtype=pred_box.dtype),
        torch.arange(output_width, device=pred_box.device, dtype=pred_box.dtype),
        indexing="ij",
    )
    pred_box = pred_box.unsqueeze(1).unsqueeze(2)
    mask = (
        (y < pred_box[..., 1])
        | (y > pred_box[..., 3])
        | (x < pred_box[..., 0])
        | (x > pred_box[..., 2])
    )
    pred_mask.masked_fill_(mask, -65504.0)

    return pred_mask
