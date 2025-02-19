import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from e2edet.module.ops import BoxAttnFunction, InstanceAttnFunction


@torch.jit.script
def box_to_grid(
    ref_windows: torch.Tensor,
    ref_windows_size: torch.Tensor,
    kernel_indices: torch.Tensor,
    offset_boxes: Optional[torch.Tensor] = None,
    normalizer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if offset_boxes is None:
        boxes = ref_windows
    else:
        if normalizer is None:
            boxes = ref_windows + offset_boxes / 8 * ref_windows_size
        else:
            boxes = ref_windows + offset_boxes * normalizer * ref_windows_size
    center, size = boxes.unsqueeze(-2).split(2, dim=-1)

    grid = center + kernel_indices * torch.relu(size)

    return grid


class InstanceAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_level,
        num_head,
        kernel_size,
        attn_size=2,
        attn_mode="all",
        threshold=0,
        normalizer=False,
    ):
        super(InstanceAttention, self).__init__()
        assert d_model % num_head == 0, "d_model should be divided by num_head"
        assert attn_mode in (
            "all",
            "sigmoid",
            "softmax",
            "sigmoid_v1",
            "sigmoid_v2",
            "sigmoid_v3",
            "sigmoid_v4",
        )
        assert kernel_size % attn_size == 0

        self.im2col_step = 64
        self.d_model = d_model
        self.num_head = num_head
        self.num_level = num_level
        self.kernel_size = kernel_size
        self.head_dim = d_model // num_head

        self.linear_box_weight = nn.Parameter(
            torch.zeros(num_level * num_head * 4, d_model)
        )
        self.linear_box_bias = nn.Parameter(torch.zeros(num_head * num_level * 4))

        self.linear_attn_weight = nn.Parameter(
            torch.zeros(num_head * num_level * (attn_size**2), d_model)
        )
        self.linear_attn_bias = nn.Parameter(
            torch.zeros(num_head * num_level * (attn_size**2))
        )

        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if normalizer:
            self.normalizer = nn.Parameter(torch.ones(1) * 0.125)
        else:
            self.normalizer = None

        self.attn_mode = attn_mode
        self.attn_size = attn_size
        self.threshold = threshold

        self._create_kernel_indices(self.kernel_size, "kernel_indices")
        self._reset_parameters()

    def _create_kernel_indices(self, kernel_size, module_name):
        if kernel_size % 2 == 0:
            start_idx = -kernel_size // 2
            end_idx = kernel_size // 2

            indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
        else:
            start_idx = -(kernel_size - 1) // 2
            end_idx = (kernel_size - 1) // 2

            indices = torch.linspace(start_idx, end_idx, kernel_size)
        i, j = torch.meshgrid(indices, indices, indexing="ij")
        kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / kernel_size
        self.register_buffer(module_name, kernel_indices)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.linear_attn_weight, 0.0)
        nn.init.constant_(self.linear_attn_bias, 0.0)
        nn.init.constant_(self.linear_box_weight, 0.0)
        nn.init.uniform_(self.linear_box_bias)

    def _where_to_attend(self, query, v_valid_ratios, ref_windows, attn_mask=None):
        b, l = ref_windows.shape[:2]

        offset_boxes = F.linear(query, self.linear_box_weight, self.linear_box_bias)
        # if self.normalizer is not None:
        #     offset_boxes = offset_boxes.view(b, l, self.num_head * self.num_level, 4)
        #     offset_u = offset_boxes.mean(2, keepdim=True)
        #     offset_s = (offset_boxes - offset_u).pow(2).mean(2, keepdim=True)
        #     offset_boxes = (offset_boxes - offset_u) / torch.sqrt(offset_s + 1e-6)

        offset_boxes = offset_boxes.view(b, l, self.num_head, self.num_level, 4)

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
        else:
            ref_windows = ref_windows.unsqueeze(3)

        ref_windows_size = ref_windows[..., [2, 3, 2, 3]]
        grid = box_to_grid(
            ref_windows,
            ref_windows_size,
            self.kernel_indices,
            offset_boxes,
            self.normalizer,
        )

        # grid: (B, L, nhead, nlevel, npoint, 2)
        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios
        grid = grid.contiguous()

        if attn_mask is not None:
            attn_mask = F.grid_sample(
                attn_mask.flatten(0, 1)[:, None],
                2.0 * grid.view(b * l, self.num_head, -1, 2) - 1.0,
                align_corners=False,
            )

            if self.attn_mode == "all":
                attn_mask = (attn_mask.sigmoid() < self.threshold).view(
                    b, l, self.num_head, -1
                )
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            elif self.attn_mode == "sigmoid":
                attn_mask = attn_mask.sigmoid()
            elif self.attn_mode == "sigmoid_v1":
                attn_mask = (attn_mask / 0.1).sigmoid()
            elif self.attn_mode == "sigmoid_v2":
                attn_mask = (attn_mask / 0.01).sigmoid()
            elif self.attn_mode == "sigmoid_v3":
                attn_mask = (attn_mask / 0.5).sigmoid()
            elif self.attn_mode == "sigmoid_v4":
                attn_mask = (attn_mask / 2).sigmoid()
            elif self.attn_mode == "softmax":
                attn_mask = attn_mask.view(b, l, self.num_head, -1).softmax(dim=-1)
            attn_mask = attn_mask.view(
                b, l, self.num_head, self.num_level, self.kernel_size, self.kernel_size
            ).detach()

        return grid, attn_mask

    def forward(
        self,
        query,
        value,
        v_shape,
        v_mask,
        v_start_index,
        v_valid_ratios,
        ref_windows,
        attn_mask=None,
    ):
        b, l1 = query.shape[:2]
        l2 = value.shape[1]

        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(b, l2, self.num_head, self.head_dim)

        sampled_grid, attn_mask = self._where_to_attend(
            query, v_valid_ratios, ref_windows, attn_mask
        )

        attn_weights = F.linear(query, self.linear_attn_weight, self.linear_attn_bias)
        attn_weights = attn_weights.view(
            b, l1, self.num_head, self.num_level, self.attn_size, self.attn_size
        )
        num_repeat = self.kernel_size // self.attn_size
        attn_weights = attn_weights.repeat_interleave(num_repeat, dim=-1)
        attn_weights = attn_weights.repeat_interleave(num_repeat, dim=-2)

        if attn_mask is not None:
            if self.attn_mode == "all":
                spatial_attn_weights = attn_weights.masked_fill(attn_mask, -65504.0)
            elif self.attn_mode in (
                "sigmoid",
                "softmax",
                "sigmoid_v1",
                "sigmoid_v2",
                "sigmoid_v3",
                "sigmoid_v4",
            ):
                spatial_attn_weights = attn_weights * attn_mask
        else:
            spatial_attn_weights = attn_weights

        spatial_attn_weights = spatial_attn_weights.view(b, l1, self.num_head, -1)
        spatial_attn_weights = F.softmax(spatial_attn_weights, dim=-1).view(
            b, l1, self.num_head, self.num_level, self.kernel_size, self.kernel_size
        )

        if not self.inferencing:
            level_attn_weights = attn_weights.view(
                b, l1, self.num_head, self.num_level, self.kernel_size, self.kernel_size
            )
            level_attn_weights = F.softmax(level_attn_weights, dim=3)

            if attn_mask is not None:
                level_attn_weights = level_attn_weights.masked_fill(attn_mask, 0.0)

            output, mask_output = InstanceAttnFunction.apply(
                value,
                v_shape,
                v_start_index,
                sampled_grid,
                spatial_attn_weights,
                level_attn_weights,
                self.kernel_size,
                self.im2col_step,
            )
            mask_output = self.out_proj(mask_output)
        else:
            output = BoxAttnFunction.apply(
                value,
                v_shape,
                v_start_index,
                sampled_grid,
                spatial_attn_weights,
                self.im2col_step,
            )
            mask_output = None
        output = self.out_proj(output)

        return output, mask_output, attn_weights


class BoxAttention(nn.Module):
    def __init__(self, d_model, num_level, num_head, kernel_size=2, normalizer=False):
        super(BoxAttention, self).__init__()
        assert d_model % num_head == 0, "d_model should be divided by num_head"

        self.im2col_step = 128
        self.d_model = d_model
        self.num_head = num_head
        self.num_level = num_level
        self.head_dim = d_model // num_head
        self.kernel_size = kernel_size
        self.num_point = kernel_size**2

        self.linear_box_weight = nn.Parameter(
            torch.zeros(num_level * num_head * 4, d_model)
        )
        self.linear_box_bias = nn.Parameter(torch.zeros(num_head * num_level * 4))

        self.linear_attn_weight = nn.Parameter(
            torch.zeros(num_head * num_level * self.num_point, d_model)
        )
        self.linear_attn_bias = nn.Parameter(
            torch.zeros(num_head * num_level * self.num_point)
        )

        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if normalizer:
            self.normalizer = nn.Parameter(torch.ones(1) * 0.125)
        else:
            self.normalizer = None
        self._create_kernel_indices(kernel_size, "kernel_indices")
        self._reset_parameters()

    def _create_kernel_indices(self, kernel_size, module_name):
        if kernel_size % 2 == 0:
            start_idx = -kernel_size // 2
            end_idx = kernel_size // 2

            indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
        else:
            start_idx = -(kernel_size - 1) // 2
            end_idx = (kernel_size - 1) // 2

            indices = torch.linspace(start_idx, end_idx, kernel_size)
        i, j = torch.meshgrid(indices, indices, indexing="ij")
        kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / self.kernel_size
        self.register_buffer(module_name, kernel_indices)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.linear_attn_weight, 0.0)
        nn.init.constant_(self.linear_attn_bias, 0.0)
        nn.init.constant_(self.linear_box_weight, 0.0)
        nn.init.uniform_(self.linear_box_bias)

    def _where_to_attend(self, query, v_valid_ratios, ref_windows):
        b, l = ref_windows.shape[:2]

        offset_boxes = F.linear(query, self.linear_box_weight, self.linear_box_bias)
        # if self.normalizer is not None:
        #     offset_boxes = offset_boxes.view(b, l, self.num_head * self.num_level, 4)
        #     offset_u = offset_boxes.mean(2, keepdim=True)
        #     offset_s = (offset_boxes - offset_u).pow(2).mean(2, keepdim=True)
        #     offset_boxes = (offset_boxes - offset_u) / torch.sqrt(offset_s + 1e-6)

        offset_boxes = offset_boxes.view(b, l, self.num_head, self.num_level, 4)

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
        else:
            ref_windows = ref_windows.unsqueeze(3)

        ref_windows_size = ref_windows[..., [2, 3, 2, 3]]
        grid = box_to_grid(
            ref_windows,
            ref_windows_size,
            self.kernel_indices,
            offset_boxes,
            self.normalizer,
        )

        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios

        return grid.contiguous()

    def forward(
        self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows
    ):
        b, l1 = query.shape[:2]
        l2 = value.shape[1]

        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(b, l2, self.num_head, self.head_dim)

        attn_weights = F.linear(query, self.linear_attn_weight, self.linear_attn_bias)
        attn_weights = F.softmax(attn_weights.view(b, l1, self.num_head, -1), dim=-1)
        attn_weights = attn_weights.view(
            b, l1, self.num_head, self.num_level, self.kernel_size, self.kernel_size
        )

        sampled_grid = self._where_to_attend(query, v_valid_ratios, ref_windows)
        output = BoxAttnFunction.apply(
            value, v_shape, v_start_index, sampled_grid, attn_weights, self.im2col_step
        )
        output = self.out_proj(output)

        return output, attn_weights


class SimpleInstanceAttention(nn.Module):
    def __init__(
        self,
        q_dim,
        v_dim,
        num_box,
        num_head,
        kernel_size,
        attn_size=2,
        normalizer=False,
    ):
        super(SimpleInstanceAttention, self).__init__()
        assert v_dim % num_head == 0, "v_dim should be divided by num_head"

        self.im2col_step = 64
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.num_head = num_head
        self.kernel_size = kernel_size
        self.num_point = kernel_size**2
        self.head_dim = v_dim // num_head
        self.attn_size = attn_size
        self.num_box = num_box

        self.linear_box_weight = nn.Parameter(
            torch.zeros(num_box * num_head * 4, q_dim)
        )
        self.linear_box_bias = nn.Parameter(torch.zeros(num_head * num_box * 4))

        self.linear_attn_weight = nn.Parameter(
            torch.zeros(num_head * num_box * 4, q_dim)
        )
        self.linear_attn_bias = nn.Parameter(torch.zeros(num_head * num_box * 4))

        self.value_proj = nn.Linear(v_dim, v_dim)
        self.out_proj = nn.Linear(v_dim, q_dim)

        if normalizer:
            self.normalizer = nn.Parameter(torch.ones(1) * 0.125)
        else:
            self.normalizer = None

        self._create_kernel_indices(self.kernel_size, "kernel_indices")
        self._reset_parameters()

    def _create_kernel_indices(self, kernel_size, module_name):
        if kernel_size % 2 == 0:
            start_idx = -kernel_size // 2
            end_idx = kernel_size // 2

            indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
        else:
            start_idx = -(kernel_size - 1) // 2
            end_idx = (kernel_size - 1) // 2

            indices = torch.linspace(start_idx, end_idx, kernel_size)
        i, j = torch.meshgrid(indices, indices, indexing="ij")
        kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / self.kernel_size
        self.register_buffer(module_name, kernel_indices)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.linear_attn_weight, 0.0)
        nn.init.constant_(self.linear_attn_bias, 0.0)
        nn.init.constant_(self.linear_box_weight, 0.0)
        nn.init.uniform_(self.linear_box_bias)

    def _where_to_attend(self, query, v_valid_ratios, ref_windows, attn_mask=None):
        b, l = ref_windows.shape[:2]

        offset_boxes = F.linear(query, self.linear_box_weight, self.linear_box_bias)
        # if self.normalizer is not None:
        #     offset_boxes = offset_boxes.view(b, l, self.num_head * self.num_box, 4)
        #     offset_u = offset_boxes.mean(2, keepdim=True)
        #     offset_s = (offset_boxes - offset_u).pow(2).mean(2, keepdim=True)
        #     offset_boxes = (offset_boxes - offset_u) / torch.sqrt(offset_s + 1e-6)

        offset_boxes = offset_boxes.view(b, l, self.num_head, self.num_box, 4)

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
        else:
            ref_windows = ref_windows.unsqueeze(2)

        ref_windows_size = ref_windows[..., [2, 3, 2, 3]]
        grid = box_to_grid(
            ref_windows,
            ref_windows_size,
            self.kernel_indices,
            offset_boxes,
            self.normalizer,
        )

        # grid: (B, L, nhead, nbox, npoint, 2)
        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios
        grid = grid.contiguous()

        if attn_mask is not None:
            attn_mask = F.grid_sample(
                attn_mask.flatten(0, 1)[:, None],
                2.0 * grid.view(b * l, self.num_head, -1, 2) - 1.0,
                align_corners=False,
            )
            attn_mask = (attn_mask.sigmoid() < 0.5).view(b, l, self.num_head, -1)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            attn_mask = attn_mask.view(
                b, l, self.num_head, self.num_box, self.kernel_size, self.kernel_size
            ).detach()

        return grid, attn_mask

    def forward(
        self,
        query,
        value,
        v_shape,
        v_mask,
        v_start_index,
        v_valid_ratios,
        ref_windows,
        attn_mask=None,
    ):
        b, l1 = query.shape[:2]
        l2 = value.shape[1]

        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(b, l2, self.num_head, self.head_dim)

        sampled_grid, attn_mask = self._where_to_attend(
            query, v_valid_ratios, ref_windows, attn_mask
        )

        attn_weights = F.linear(query, self.linear_attn_weight, self.linear_attn_bias)
        attn_weights = attn_weights.view(b, l1, self.num_head, self.num_box, 2, 2)
        attn_weights = attn_weights.repeat_interleave(self.kernel_size // 2, dim=-1)
        attn_weights = attn_weights.repeat_interleave(self.kernel_size // 2, dim=-2)

        if attn_mask is not None:
            spatial_attn_weights = attn_weights.masked_fill(attn_mask, -65504.0)
        else:
            spatial_attn_weights = attn_weights

        spatial_attn_weights = spatial_attn_weights.view(b, l1, self.num_head, -1)
        spatial_attn_weights = F.softmax(spatial_attn_weights, dim=-1).view(
            b, l1, self.num_head, 1, self.num_box * self.num_point
        )
        sampled_grid = sampled_grid.view(
            b, l1, self.num_head, 1, self.num_box * self.num_point, 2
        )

        if not self.inferencing:
            level_attn_weights = attn_weights.view(
                b, l1, self.num_head, self.num_box, self.kernel_size, self.kernel_size
            )
            level_attn_weights = F.softmax(level_attn_weights, dim=3)
            level_attn_weights = level_attn_weights.view(
                b, l1, self.num_head, 1, self.num_box * self.num_point
            )

            output, mask_output = InstanceAttnFunction.apply(
                value,
                v_shape,
                v_start_index,
                sampled_grid,
                spatial_attn_weights,
                level_attn_weights,
                (self.num_box, self.kernel_size, self.kernel_size),
                self.im2col_step,
            )
            mask_output = self.out_proj(mask_output.sum(2))
        else:
            output = BoxAttnFunction.apply(
                value,
                v_shape,
                v_start_index,
                sampled_grid,
                spatial_attn_weights,
                self.im2col_step,
            )
            mask_output = None
        output = self.out_proj(output)

        return output, mask_output, attn_weights


class SimpleBoxAttention(nn.Module):
    def __init__(
        self, q_dim, v_dim, num_box, num_head, kernel_size=2, normalizer=False
    ):
        super(SimpleBoxAttention, self).__init__()
        assert (
            v_dim % num_head == 0
        ), f"v_dim ({v_dim}) should be divided by num_head ({num_head})"

        self.im2col_step = 64
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.num_head = num_head
        self.num_box = num_box
        self.kernel_size = kernel_size
        self.num_point = kernel_size**2
        self.head_dim = v_dim // num_head

        self.linear_box_weight = nn.Parameter(
            torch.zeros(num_box * num_head * 4, q_dim)
        )
        self.linear_box_bias = nn.Parameter(torch.zeros(num_head * num_box * 4))

        self.linear_attn_weight = nn.Parameter(
            torch.zeros(num_head * num_box * self.num_point, q_dim)
        )
        self.linear_attn_bias = nn.Parameter(
            torch.zeros(num_head * num_box * self.num_point)
        )

        self.value_proj = nn.Linear(v_dim, v_dim)
        self.out_proj = nn.Linear(v_dim, q_dim)

        if normalizer:
            self.normalizer = nn.Parameter(torch.ones(1) * 0.125)
        else:
            self.normalizer = None

        self._create_kernel_indices(kernel_size, "kernel_indices")
        self._reset_parameters()

    def _create_kernel_indices(self, kernel_size, module_name):
        if kernel_size % 2 == 0:
            start_idx = -kernel_size // 2
            end_idx = kernel_size // 2

            indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
        else:
            start_idx = -(kernel_size - 1) // 2
            end_idx = (kernel_size - 1) // 2

            indices = torch.linspace(start_idx, end_idx, kernel_size)
        i, j = torch.meshgrid(indices, indices, indexing="ij")
        kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / self.kernel_size
        self.register_buffer(module_name, kernel_indices)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.linear_attn_weight, 0.0)
        nn.init.constant_(self.linear_attn_bias, 0.0)
        nn.init.constant_(self.linear_box_weight, 0.0)
        nn.init.uniform_(self.linear_box_bias)

    def _where_to_attend(self, query, v_valid_ratios, ref_windows):
        b, l = ref_windows.shape[:2]

        offset_boxes = F.linear(query, self.linear_box_weight, self.linear_box_bias)
        # if self.normalizer is not None:
        #     offset_boxes = offset_boxes.view(b, l, self.num_head * self.num_box, 4)
        #     offset_u = offset_boxes.mean(2, keepdim=True)
        #     offset_s = (offset_boxes - offset_u).pow(2).mean(2, keepdim=True)
        #     offset_boxes = (offset_boxes - offset_u) / torch.sqrt(offset_s + 1e-6)

        offset_boxes = offset_boxes.view(b, l, self.num_head, self.num_box, 4)

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
        else:
            ref_windows = ref_windows.unsqueeze(2)

        ref_windows_size = ref_windows[..., [2, 3, 2, 3]]
        grid = box_to_grid(
            ref_windows,
            ref_windows_size,
            self.kernel_indices,
            offset_boxes,
            self.normalizer,
        )
        # self.boxes = boxes

        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios

        return grid.contiguous()

    def forward(
        self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows
    ):
        b, l1 = query.shape[:2]
        l2 = value.shape[1]

        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(b, l2, self.num_head, self.head_dim)

        attn_weights = F.linear(query, self.linear_attn_weight, self.linear_attn_bias)
        attn_weights = F.softmax(attn_weights.view(b, l1, self.num_head, -1), dim=-1)
        attn_weights = attn_weights.view(
            b, l1, self.num_head, 1, self.num_box * self.num_point
        )
        # self.attn = attn_weights.view(
        #     b, l1, self.num_head, self.num_box, self.num_point
        # )

        sampled_grid = self._where_to_attend(query, v_valid_ratios, ref_windows)
        sampled_grid = sampled_grid.view(
            b, l1, self.num_head, 1, self.num_box * self.num_point, 2
        )

        output = BoxAttnFunction.apply(
            value, v_shape, v_start_index, sampled_grid, attn_weights, self.im2col_step
        )
        output = self.out_proj(output)

        return output, attn_weights


class Box3dAttention(nn.Module):
    def __init__(self, d_model, num_level, num_head, with_rotation=True, kernel_size=2):
        super(Box3dAttention, self).__init__()
        assert d_model % num_head == 0, "d_model should be divided by num_head"

        num_variable = 5 if with_rotation else 4

        self.im2col_step = 128
        self.d_model = d_model
        self.num_head = num_head
        self.num_level = num_level
        self.head_dim = d_model // num_head
        self.with_rotation = with_rotation
        self.num_variable = num_variable
        self.kernel_size = kernel_size
        self.num_point = kernel_size**2

        self.linear_box_weight = nn.Parameter(
            torch.zeros(num_level * num_head * num_variable, d_model)
        )
        self.linear_box_bias = nn.Parameter(
            torch.zeros(num_head * num_level * num_variable)
        )

        self.linear_attn_weight = nn.Parameter(
            torch.zeros(num_head * num_level * self.num_point, d_model)
        )
        self.linear_attn_bias = nn.Parameter(
            torch.zeros(num_head * num_level * self.num_point)
        )

        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self._create_kernel_indices(kernel_size, "kernel_indices")
        self._reset_parameters()

    def _create_kernel_indices(self, kernel_size, module_name):
        if kernel_size % 2 == 0:
            start_idx = -kernel_size // 2
            end_idx = kernel_size // 2

            indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
        else:
            start_idx = -(kernel_size - 1) // 2
            end_idx = (kernel_size - 1) // 2

            indices = torch.linspace(start_idx, end_idx, kernel_size)
        i, j = torch.meshgrid(indices, indices, indexing="ij")
        kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / 2
        self.register_buffer(module_name, kernel_indices)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.linear_attn_weight, 0.0)
        nn.init.constant_(self.linear_attn_bias, 0.0)
        nn.init.constant_(self.linear_box_weight, 0.0)
        nn.init.uniform_(self.linear_box_bias)

    def _where_to_attend(self, query, v_valid_ratios, ref_windows):
        b, l = ref_windows.shape[:2]

        offset_boxes = F.linear(query, self.linear_box_weight, self.linear_box_bias)
        offset_boxes = offset_boxes.view(
            b, l, self.num_head, self.num_level, self.num_variable
        )

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
            ref_windows, ref_angles, _ = ref_windows.split((4, 1, 2), dim=-1)
        else:
            ref_windows = ref_windows.unsqueeze(3)
            ref_windows, ref_angles = ref_windows.split((4, 1), dim=-1)

        if self.with_rotation:
            offset_boxes, offset_angles = offset_boxes.split(4, dim=-1)
            angles = (ref_angles + offset_angles / 16) * 2 * math.pi
        else:
            angles = ref_angles.expand(b, l, self.num_head, self.num_level, 1)

        boxes = ref_windows + offset_boxes / 8 * ref_windows[..., [2, 3, 2, 3]]
        center, size = boxes.unsqueeze(-2).split(2, dim=-1)

        cos_angle, sin_angle = torch.cos(angles), torch.sin(angles)
        rot_matrix = torch.stack([cos_angle, -sin_angle, sin_angle, cos_angle], dim=-1)
        rot_matrix = rot_matrix.view(b, l, self.num_head, self.num_level, 1, 2, 2)

        grid = self.kernel_indices * torch.relu(size)
        grid = center + (grid.unsqueeze(-2) * rot_matrix).sum(-1)

        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios

        return grid.contiguous()

    def forward(
        self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows
    ):
        b, l1 = query.shape[:2]
        l2 = value.shape[1]

        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(b, l2, self.num_head, self.head_dim)

        attn_weights = F.linear(query, self.linear_attn_weight, self.linear_attn_bias)
        attn_weights = F.softmax(attn_weights.view(b, l1, self.num_head, -1), dim=-1)
        attn_weights = attn_weights.view(
            b, l1, self.num_head, self.num_level, self.kernel_size, self.kernel_size
        )

        sampled_grid = self._where_to_attend(query, v_valid_ratios, ref_windows)
        output = BoxAttnFunction.apply(
            value, v_shape, v_start_index, sampled_grid, attn_weights, self.im2col_step
        )
        output = self.out_proj(output)

        return output, attn_weights
