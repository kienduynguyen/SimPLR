import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from e2edet.module.ops import BoxAttnFunction
from e2edet.utils.distributed import synchronize


class DeformableAttention(nn.Module):
    def __init__(self, q_dim, v_dim, num_level, num_head, num_point=4):
        super(DeformableAttention, self).__init__()
        assert v_dim % num_head == 0, "d_model should be divided by num_head"

        self.im2col_step = 64
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.num_head = num_head
        self.num_level = num_level
        self.num_point = num_point
        self.head_dim = v_dim // num_head

        self.sampling_offsets_weight = nn.Parameter(
            torch.zeros(num_head * num_level * num_point * 2, q_dim)
        )
        self.sampling_offsets_bias = nn.Parameter(
            torch.zeros(num_head * num_level * num_point * 2)
        )
        self.attention_weights = nn.Linear(q_dim, num_head * num_level * num_point)
        self.value_proj = nn.Linear(v_dim, v_dim)
        self.out_proj = nn.Linear(v_dim, q_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets_weight, 0.0)
        thetas = torch.arange(self.num_head, dtype=torch.float32) * (
            2.0 * math.pi / self.num_head
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_head, 1, 1, 2)
            .repeat(1, self.num_level, self.num_point, 1)
        )
        for i in range(self.num_point):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets_bias.data.copy_(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_normal_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows
    ):
        b, l1 = query.shape[:2]
        l2 = value.shape[1]

        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(b, l2, self.num_head, self.head_dim)

        sampling_offsets = F.linear(
            query, self.sampling_offsets_weight, self.sampling_offsets_bias
        ).view(b, l1, self.num_head, self.num_level, self.num_point, 2)
        attention_weights = self.attention_weights(query).view(
            b, l1, self.num_head, self.num_level * self.num_point
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            b, l1, self.num_head, self.num_level, self.num_point
        )
        sampling_locations = (
            ref_windows[:, :, None, None, None, :2]
            + sampling_offsets / 8 * ref_windows[:, :, None, None, None, 2:]
        )
        if v_valid_ratios is not None:
            sampling_locations = sampling_locations * v_valid_ratios

        output = BoxAttnFunction.apply(
            value,
            v_shape,
            v_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        output = self.out_proj(output)

        return output, attention_weights


class ScaleAwareDeformableAttention(nn.Module):
    def __init__(self, q_dim, v_dim, num_scale, num_head, num_point=4):
        super(ScaleAwareDeformableAttention, self).__init__()
        assert v_dim % num_head == 0, "d_model should be divided by num_head"

        self.im2col_step = 64
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.num_head = num_head
        self.num_scale = num_scale
        self.num_point = num_point
        self.head_dim = v_dim // num_head

        self.sampling_offsets_weight = nn.Parameter(
            torch.zeros(num_head * num_scale * num_point * 2, q_dim)
        )
        self.sampling_offsets_bias = nn.Parameter(
            torch.zeros(num_head * num_scale * num_point * 2)
        )
        self.attention_weights = nn.Linear(q_dim, num_head * num_scale * num_point)
        self.value_proj = nn.Linear(v_dim, v_dim)
        self.out_proj = nn.Linear(v_dim, q_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets_weight, 0.0)
        thetas = torch.arange(self.num_head * self.num_point, dtype=torch.float32) * (
            2.0 * math.pi / (self.num_head * self.num_point)
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1).view(
            self.num_head, 1, self.num_point, 2
        )
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).repeat(
            1, self.num_scale, 1, 1
        )

        self.sampling_offsets_bias.data.copy_(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_normal_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_normal_(self.out_proj.weight.data)
        nn.init.constant_(self.out_proj.bias.data, 0.0)

    def forward(
        self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows
    ):
        b, l1 = query.shape[:2]
        l2 = value.shape[1]

        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(b, l2, self.num_head, self.head_dim)

        sampling_offsets = F.linear(
            query, self.sampling_offsets_weight, self.sampling_offsets_bias
        ).view(b, l1, self.num_head, self.num_scale, self.num_point, 2)

        attention_weights = self.attention_weights(query).view(
            b, l1, self.num_head, self.num_scale * self.num_point
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            b, l1, self.num_head, 1, self.num_scale * self.num_point
        )

        if ref_windows.dim() == 4:
            ref_windows = ref_windows[:, :, None, :, None, :]
        elif ref_windows.dim() == 3:
            ref_windows = ref_windows[:, :, None, None, None, :]
        else:
            raise RuntimeError("ref_windows should have 3 or 4 dimensions")
        sampling_locations = (
            ref_windows[..., :2] + sampling_offsets / 8 * ref_windows[..., 2:]
        )
        # print("ref_windows:", ref_windows.shape)
        # synchronize()
        sampling_locations = sampling_locations.view(
            b, l1, self.num_head, 1, self.num_scale * self.num_point, 2
        )
        if v_valid_ratios is not None:
            sampling_locations = sampling_locations * v_valid_ratios

        output = BoxAttnFunction.apply(
            value,
            v_shape,
            v_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        output = self.out_proj(output)

        return output, attention_weights
