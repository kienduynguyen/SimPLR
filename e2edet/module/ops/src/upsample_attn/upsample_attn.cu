#include "upsample_attn_kernel.cuh"

#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace e2edet {

at::Tensor upsample_attn_cuda_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &level_attn_weight,
    const int im2col_step
)
{
    CHECK_INPUT(value);
    CHECK_INPUT(spatial_shapes);
    CHECK_INPUT(level_start_index);
    CHECK_INPUT(sampling_loc);
    CHECK_INPUT(level_attn_weight);

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto mask_output = at::zeros({batch, num_query, num_point, num_heads, channels}, value.options());

    const int batch_n = im2col_step_;
    auto mask_output_n = mask_output.view({batch / im2col_step_, batch_n, num_query, num_point, num_heads, channels});

    const int per_value_size = spatial_size * num_heads * channels;
    const int per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    const int per_sample_loc_size = per_attn_weight_size << 1;

    for (int n = 0; n < batch / im2col_step_; ++n) 
    {
        auto mask_output_columns = mask_output_n.select(0, n);

        AT_DISPATCH_FLOATING_TYPES(value.type(), "upsample_attn_forward_cuda", ( [&] {
            upsample_attn_im2col_cuda(
                at::cuda::getCurrentCUDAStream(),
                value.data<scalar_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.data<int64_t>(),
                level_start_index.data<int64_t>(),
                sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                level_attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                mask_output_columns.data<scalar_t>()
            );
        }));
    }

    mask_output = mask_output.view({batch, num_query, num_point, num_heads * channels});

    return mask_output;
}


std::vector<at::Tensor> upsample_attn_cuda_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &level_attn_weight,
    const at::Tensor &grad_mask_output,
    const int im2col_step
)
{
    CHECK_INPUT(value);
    CHECK_INPUT(spatial_shapes);
    CHECK_INPUT(level_start_index);
    CHECK_INPUT(sampling_loc);
    CHECK_INPUT(level_attn_weight);
    CHECK_INPUT(grad_mask_output);

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_level_attn_weight = at::zeros_like(level_attn_weight);
    
    const int batch_n = im2col_step_;
    const int per_value_size = spatial_size * num_heads * channels;
    const int per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    const int per_sample_loc_size = per_attn_weight_size << 1;
    auto grad_mask_output_n = grad_mask_output.view({batch / im2col_step_, batch_n, num_query, num_point, num_heads, channels});

    for (int n = 0; n < batch / im2col_step_; ++n) 
    {
        auto grad_mask_output_columns = grad_mask_output_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES(value.type(), "upsample_attn_backward_cuda", ( [&] {
            upsample_attn_col2im_cuda(
                at::cuda::getCurrentCUDAStream(),
                grad_mask_output_columns.data<scalar_t>(),
                value.data<scalar_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.data<int64_t>(),
                level_start_index.data<int64_t>(),
                sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                level_attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                grad_value.data<scalar_t>() + n * im2col_step_ * per_value_size,
                grad_sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                grad_level_attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size
            );
        }));
    }

    return {grad_value, grad_sampling_loc, grad_level_attn_weight};
}

}