import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.amp import custom_fwd, custom_bwd

from e2edet import ops


class BoxAttnFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = ops.box_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )

        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = ops.box_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class InstanceAttnFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        spatial_attention_weights,
        level_attention_weights,
        mask_size,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output, mask_output = ops.instance_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            spatial_attention_weights,
            level_attention_weights,
            im2col_step,
        )

        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            spatial_attention_weights,
            level_attention_weights,
        )

        b, l, _, c = mask_output.shape
        if isinstance(mask_size, int):
            mask_output = mask_output.view(b, l, mask_size, mask_size, c)
        else:
            shape = [b, l] + list(mask_size) + [c]
            mask_output = mask_output.view(*shape)

        return output, mask_output

    @staticmethod
    @custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output, grad_mask_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            spatial_attention_weights,
            level_attention_weights,
        ) = ctx.saved_tensors

        (
            grad_value,
            grad_sampling_loc,
            grad_spatial_attn_weight,
            grad_level_attn_weight,
        ) = ops.instance_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            spatial_attention_weights,
            level_attention_weights,
            grad_output.contiguous(),
            grad_mask_output.contiguous(),
            ctx.im2col_step,
        )

        return (
            grad_value,
            None,
            None,
            grad_sampling_loc,
            grad_spatial_attn_weight,
            grad_level_attn_weight,
            None,
            None,
        )


class FastBoxAttnFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = ops.fast_box_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )

        return output

    @staticmethod
    @custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = ops.fast_box_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class FastInstanceAttnFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        spatial_attention_weights,
        level_attention_weights,
        mask_size,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output, mask_output = ops.fast_instance_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            spatial_attention_weights,
            level_attention_weights,
            im2col_step,
        )

        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            spatial_attention_weights,
            level_attention_weights,
        )

        b, l, _, c = mask_output.shape
        mask_output = mask_output.view(b, l, mask_size, mask_size, c)

        return output, mask_output

    @staticmethod
    @custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx, grad_output, grad_mask_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            spatial_attention_weights,
            level_attention_weights,
        ) = ctx.saved_tensors

        (
            grad_value,
            grad_sampling_loc,
            grad_spatial_attn_weight,
            grad_level_attn_weight,
        ) = ops.fast_instance_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            spatial_attention_weights,
            level_attention_weights,
            grad_output.contiguous(),
            grad_mask_output.contiguous(),
            ctx.im2col_step,
        )

        return (
            grad_value,
            None,
            None,
            grad_sampling_loc,
            grad_spatial_attn_weight,
            grad_level_attn_weight,
            None,
            None,
        )
