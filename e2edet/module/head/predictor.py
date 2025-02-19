import torch
import torch.nn as nn
import torch.nn.functional as F

from e2edet.utils.general import inverse_sigmoid
from e2edet.utils.distributed import synchronize

from .helper import MLP, SegmentMLP, SegmentMLPv2


class MultiDetector(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_references,
        loss_mode="focal",
        mode_pred="detr",
    ):
        super(MultiDetector, self).__init__()
        assert mode_pred == "detr"

        self.num_references = num_references
        self.mode_pred = mode_pred

        if loss_mode in ("focal", "bce"):
            self.class_embed = nn.Linear(hidden_dim, num_references * num_classes)
        else:
            self.class_embed = nn.Linear(hidden_dim, num_references * (num_classes + 1))

        self.bbox_embed = MLP(hidden_dim, hidden_dim, num_references * 4, 3)

    def forward(self, x, ref_windows=None, x_mask=None):
        n, b, l = x.shape[:3]
        ref_windows = ref_windows[..., : self.num_references, :]

        outputs_class = self.class_embed(x).view(n, b, l, self.num_references, -1)

        if x_mask is not None:
            outputs_class = outputs_class.masked_fill(
                x_mask.unsqueeze(-1).unsqueeze(-1), -65504.0
            )
        outputs_class = outputs_class.view(n, b, l * self.num_references, -1)

        outputs_coord = self.bbox_embed(x).view(n, b, l, self.num_references, 4)
        outputs_coord = outputs_coord + inverse_sigmoid(ref_windows)
        outputs_coord = outputs_coord.view(n, b, l * self.num_references, 4).sigmoid()

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        return out


class SimpleHead(nn.Module):
    def __init__(
        self, hidden_dim, patch_size, num_classes, mask_layer=1, loss_mode="focal"
    ):
        super(SimpleHead, self).__init__()

        if loss_mode in ("focal", "bce"):
            self.class_embed = nn.Linear(hidden_dim, num_classes)
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        self.mask_embed = MLP(
            hidden_dim,
            hidden_dim,
            patch_size**2,
            mask_layer,
        )
        self.patch_size = patch_size

    def forward(self, x, seg, seg_mask, x_mask=None):
        b, l, h, w = seg.shape[:-1]
        outputs_class = self.class_embed(x)

        if x_mask is not None:
            outputs_class = outputs_class.masked_fill(x_mask.unsqueeze(-1), -65504.0)
        out = {"pred_logits": outputs_class}

        outputs_mask = (
            self.mask_embed(seg)
            .view(b, l, h, w, self.patch_size, self.patch_size)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(b, l, h * self.patch_size, w * self.patch_size)
        )

        out["pred_masks"] = outputs_mask
        out["padding_masks"] = (
            F.interpolate(
                seg_mask.view(b, 1, h, w).float(), size=outputs_mask.shape[-2:]
            )
            .to(seg_mask)
            .repeat(1, l, 1, 1)
        )

        return out


class Segmentor(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mask_dim,
        num_classes,
        mask_mode="none",
        loss_mode="focal",
        with_box=True,
        mode_pred="detr",
    ):
        super(Segmentor, self).__init__()
        assert mask_mode in ("none", "mask_v1", "mask_v2", "proposal_v1", "proposal_v2")
        assert mode_pred in ("detr", "none", "boxer")

        if loss_mode in ("focal", "bce"):
            self.class_embed = nn.Linear(hidden_dim, num_classes)
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        if with_box:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if mask_mode == "mask_v1":
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        elif mask_mode == "mask_v2":
            self.mask_embed = nn.Linear(hidden_dim, mask_dim)
        elif mask_mode == "proposal_v1":
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        elif mask_mode == "proposal_v2":
            self.mask_embed = nn.Linear(hidden_dim, mask_dim)

        self.with_box = with_box
        self.mask_mode = mask_mode
        self.mode_pred = mode_pred

    def forward(self, x, ref_windows=None, x_mask=None, feat=None, feat_mask=None):
        outputs_class = self.class_embed(x)

        if x_mask is not None:
            outputs_class = outputs_class.masked_fill(x_mask.unsqueeze(-1), -65504.0)
        out = {"pred_logits": outputs_class}

        if self.with_box:
            outputs_coord = self.bbox_embed(x)

            if self.mode_pred == "detr":
                if ref_windows is not None:
                    if ref_windows.shape[-1] == 4:
                        outputs_coord = outputs_coord + inverse_sigmoid(ref_windows)
                    else:
                        raise ValueError("ref_windows should be 4 dim")
                out["pred_boxes"] = outputs_coord.sigmoid()
            elif self.mode_pred == "none":
                if ref_windows is not None:
                    if ref_windows.shape[-1] == 4:
                        outputs_coord += ref_windows
                    else:
                        raise ValueError("ref_windows should be 4 dim")
                out["pred_boxes"] = outputs_coord
            elif self.mode_pred == "boxer":
                if ref_windows is not None:
                    if ref_windows.shape[-1] == 4:
                        outputs_coord = (1 + torch.tanh(outputs_coord)) * ref_windows
                    else:
                        raise ValueError("ref_windows should be 4 dim")
                out["pred_boxes"] = outputs_coord

        if self.mask_mode in ("mask_v1", "mask_v2", "proposal_v1", "proposal_v2"):
            mask_embed = self.mask_embed(x)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, feat)

            l = outputs_mask.shape[1]
            out["pred_masks"] = outputs_mask
            out["padding_masks"] = (
                None
                if feat_mask is None
                else feat_mask.unsqueeze(1).repeat(1, l, 1, 1)
            )

        return out


class Detector(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_classes,
        mask_mode="none",
        loss_mode="focal",
        mode_pred="detr",
    ):
        super(Detector, self).__init__()
        assert mask_mode in ("none", "mask_v1", "mask_v2")
        assert mode_pred in ("detr", "none", "boxer")

        self.mode_pred = mode_pred
        self.mask_mode = mask_mode

        if loss_mode in ("focal", "bce"):
            self.class_embed = nn.Linear(hidden_dim, num_classes)
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if mask_mode == "mask_v1":
            self.mask_embed = SegmentMLP(hidden_dim, 256, num_classes, 2, kernel_size=1)
        elif mask_mode == "mask_v2":
            self.mask_embed = SegmentMLPv2(hidden_dim, 256, num_classes)

    def forward(self, x, ref_windows=None, roi=None, x_mask=None):
        """
        pred_logits: [batch_size x num_queries x (num_classes + 1)]
            the classification logits (including no-object) for all queries.
        pred_boxes: The normalized boxes coordinates for all queries, represented as
                    (center_x, center_y, width, height). These values are normalized in [0, 1],
                    relative to the size of each individual image (disregarding possible padding).
        """
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x)

        if self.mask_mode == "mask_v1":
            assert roi is not None, "roi should not be None!"

            outputs_mask = self.mask_embed(roi)
            top_labels = torch.max(outputs_class, dim=-1, keepdim=True)[1]

            mask_size = outputs_mask.shape[-2]
            top_labels = (
                top_labels.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, 1, mask_size, mask_size)
            )
            outputs_mask = torch.gather(outputs_mask, 2, top_labels).squeeze(2)
        elif self.mask_mode == "mask_v2":
            assert roi is not None, "roi should not be None!"

            outputs_mask = self.mask_embed(roi)
            top_labels = torch.max(outputs_class, dim=-1, keepdim=True)[1]

            mask_size = outputs_mask.shape[-2]
            top_labels = (
                top_labels.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, 1, mask_size, mask_size)
            )
            outputs_mask = torch.gather(outputs_mask, 2, top_labels).squeeze(2)
        else:
            outputs_mask = None

        if x_mask is not None:
            outputs_class = outputs_class.masked_fill(x_mask.unsqueeze(-1), -65504.0)

        if self.mode_pred == "detr":
            if ref_windows is not None:
                if ref_windows.shape[-1] == 4:
                    outputs_coord = outputs_coord + inverse_sigmoid(ref_windows)
                else:
                    raise ValueError("ref_windows should be 4 dim")

            outputs_coord = outputs_coord.sigmoid()
        elif self.mode_pred == "none":
            outputs_coord += ref_windows
        elif self.mode_pred == "boxer":
            outputs_coord = (1 + torch.tanh(outputs_coord)) * ref_windows

        if self.mask_mode != "none":
            out = {
                "pred_logits": outputs_class,
                "pred_boxes": outputs_coord,
                "pred_masks": outputs_mask,
            }
        else:
            out = {"pred_logits": outputs_class, "pred_boxes": outputs_coord}

        return out


class Detectorv2(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_classes,
        mask_mode="none",
        loss_mode="focal",
        mode_pred="detr",
    ):
        super(Detectorv2, self).__init__()
        assert mask_mode in ("none", "mask_v1", "mask_v2")
        assert mode_pred in ("detr", "none", "boxer")

        self.mode_pred = mode_pred
        self.mask_mode = mask_mode

        if loss_mode in ("focal", "bce"):
            self.class_embed = nn.Linear(hidden_dim, num_classes)
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if mask_mode == "mask_v1":
            self.mask_embed = SegmentMLP(hidden_dim, 256, num_classes, 2, kernel_size=1)
        elif mask_mode == "mask_v2":
            self.mask_embed = SegmentMLPv2(hidden_dim, 256, 3)

    def forward(self, x, ref_windows=None, roi=None, x_mask=None):
        """
        pred_logits: [batch_size x num_queries x (num_classes + 1)]
            the classification logits (including no-object) for all queries.
        pred_boxes: The normalized boxes coordinates for all queries, represented as
                    (center_x, center_y, width, height). These values are normalized in [0, 1],
                    relative to the size of each individual image (disregarding possible padding).
        """
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x)

        if self.mask_mode == "mask_v1":
            assert roi is not None, "roi should not be None!"

            outputs_mask = self.mask_embed(roi)
            top_values = torch.max(outputs_class, dim=-1, keepdim=True)[0]
            top_mask = (top_values == outputs_class).detach()
            outputs_mask = (outputs_mask * top_mask.unsqueeze(-1).unsqueeze(-1)).sum(
                dim=2
            ) / top_mask.sum(dim=2).unsqueeze(-1).unsqueeze(-1)
        elif self.mask_mode == "mask_v2":
            assert roi is not None, "roi should not be None!"

            outputs_mask = self.mask_embed(x, roi)
        else:
            outputs_mask = None

        if x_mask is not None:
            outputs_class = outputs_class.masked_fill(x_mask.unsqueeze(-1), -65504.0)

        if self.mode_pred == "detr":
            if ref_windows is not None:
                if ref_windows.shape[-1] == 4:
                    outputs_coord = outputs_coord + inverse_sigmoid(ref_windows)
                else:
                    raise ValueError("ref_windows should be 4 dim")

            outputs_coord = outputs_coord.sigmoid()
        elif self.mode_pred == "none":
            outputs_coord += ref_windows
        elif self.mode_pred == "boxer":
            outputs_coord = (1 + torch.tanh(outputs_coord)) * ref_windows

        if self.mask_mode != "none":
            out = {
                "pred_logits": outputs_class,
                "pred_boxes": outputs_coord,
                "pred_masks": outputs_mask,
            }
        else:
            out = {"pred_logits": outputs_class, "pred_boxes": outputs_coord}

        return out


class Detector3d(nn.Module):
    def __init__(self, hidden_dim, num_classes, aux_loss):
        super(Detector3d, self).__init__()
        self.aux_loss = aux_loss

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 7, 3)

    def forward(self, x, ref_windows=None, x_mask=None):
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x) + inverse_sigmoid(ref_windows)
        outputs_coord = outputs_coord[..., [0, 1, 5, 2, 3, 6, 4]].sigmoid()

        if x_mask is not None:
            outputs_class = outputs_class.masked_fill(x_mask.unsqueeze(-1), -65504.0)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class MultiDetector3d(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_references, aux_loss):
        super(MultiDetector3d, self).__init__()
        self.aux_loss = aux_loss
        self.num_references = num_references
        self.class_embed = nn.Linear(hidden_dim, num_references * num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, num_references * 7, 3)

    def forward(self, x, ref_windows=None, x_mask=None):
        """
        pred_logits: [batch_size x num_queries x (num_classes + 1)]
            the classification logits (including no-object) for all queries.
        pred_boxes: The normalized boxes coordinates for all queries, represented as
                    (center_x, center_y, width, height). These values are normalized in [0, 1],
                    relative to the size of each individual image (disregarding possible padding).
        """
        nl, b, l = x.shape[:3]
        ref_windows = ref_windows[..., : self.num_references, :]

        outputs_class = self.class_embed(x).view(nl, b, l, self.num_references, -1)
        outputs_coord = self.bbox_embed(x).view(nl, b, l, self.num_references, 7)

        if ref_windows is not None:
            if ref_windows.shape[-1] == 5:
                outputs_box, outputs_height = outputs_coord.split((5, 2), dim=-1)
                outputs_box = outputs_box + inverse_sigmoid(ref_windows)
                outputs_coord = torch.cat([outputs_box, outputs_height], dim=-1)
                outputs_coord = outputs_coord[..., [0, 1, 5, 2, 3, 6, 4]].contiguous()
            else:
                raise ValueError("ref_windows should be 4 dim")

        if x_mask is not None:
            outputs_class = outputs_class.masked_fill(x_mask.unsqueeze(-1), -65504.0)
        outputs_class = outputs_class.view(nl, b, l * self.num_references, -1)
        outputs_coord = outputs_coord.view(nl, b, l * self.num_references, -1).sigmoid()

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
