import os
import sys
import math
import copy
import collections
import re
import warnings
import gc
import time
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch._utils import _get_available_device_type, _get_device_module

from e2edet.utils.distributed import get_world_size, synchronize
from e2edet.utils.box_ops import box_cxcywh_to_xyxy

from .logger import logger


string_classes = str


def has_cuda_capability(major: int, minor: int) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


def get_device_info():
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # default device_type: cuda
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()


# hardcoded BF16 type peak flops for NVIDIA A100, H100, H200, B200 GPU and AMD MI250, MI300X, AMD MI325X and Intel PVC
def get_peak_flops(device_name: str) -> int:
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    elif "B200" in device_name:
        # data from https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
        return 4.5e15
    elif "MI300X" in device_name or "MI325X" in device_name:
        # MI300X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
        # MI325X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
        return 1300e12
    elif "MI250X" in device_name:
        # data from https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html (per GCD)
        return 191.5e12
    elif "Data Center GPU Max 1550" in device_name:
        # Also known as Ponte Vecchio (PVC).
        # data from https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
        # Dot Product Accumulate Systolic (DPAS):
        # - Freq: 1300MHz
        # - #ops: 512
        # Full EU mode (i.e. 512 max compute units): 340.8 TFLOPS (BF16)
        # Standard EU mode (i.e. 448 max compute units): 298.2 TFLOPS (BF16)
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6
    elif "l40s" in device_name:
        # data from: "https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413"
        return 362e12

    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12


def check_if_feature_in_pytorch(
    feature_name: str,
    pull_request: str,
    min_nightly_version=None,
) -> None:
    if "git" in torch.__version__:  # pytorch is built from source
        # notify users to check if the pull request is included in their pytorch
        logger.warning(
            "detected that the pytorch is built from source. Please make sure the PR "
            f"({pull_request}) is included in pytorch for correct {feature_name}."
        )
    elif min_nightly_version is not None and torch.__version__ < min_nightly_version:
        logger.warning(
            f"detected that the pytorch version {torch.__version__} is older than "
            f"{min_nightly_version}. Please upgrade a newer version to include the "
            f"change in ({pull_request}) for correct {feature_name}."
        )


def get_dataset_absolute_path(paths):
    if os.environ.get("E2E_DATASETS") is None:
        warnings.warn("E2E_DATASETS environment not found! Setting to '.data' ...")
        os.environ["E2E_DATASETS"] = get_cache_dir(".data")

    if isinstance(paths, str):
        if not os.path.isabs(paths):
            paths = os.path.join(os.environ.get("E2E_DATASETS"), paths)
        return paths
    elif isinstance(paths, collections.Sequence):
        return [get_dataset_absolute_path(path) for path in paths]
    else:
        raise TypeError("Paths passed to dataset should either be " "string or list")


def create_ref_windows(tensor_list, mask_list, ref_size, ref_size_ratios=None):
    ref_windows = []

    eps = 1e-6
    for i, tensor in enumerate(tensor_list):
        if mask_list is not None:
            not_mask = ~(mask_list[i])
            y_embed = not_mask.cumsum(1, dtype=tensor.dtype)
            x_embed = not_mask.cumsum(2, dtype=tensor.dtype)

            size_h = not_mask[:, :, 0].sum(dim=-1, dtype=tensor.dtype)
            size_w = not_mask[:, 0, :].sum(dim=-1, dtype=tensor.dtype)
        else:
            size_h, size_w = tensor.shape[-2:]
            y_embed = torch.arange(
                1, size_h + 1, dtype=tensor.dtype, device=tensor.device
            )
            x_embed = torch.arange(
                1, size_w + 1, dtype=tensor.dtype, device=tensor.device
            )
            y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing="ij")
            x_embed = x_embed.unsqueeze(0).repeat(tensor.shape[0], 1, 1)
            y_embed = y_embed.unsqueeze(0).repeat(tensor.shape[0], 1, 1)

            size_h = torch.tensor(
                [size_h] * tensor.shape[0], dtype=tensor.dtype, device=tensor.device
            )
            size_w = torch.tensor(
                [size_w] * tensor.shape[0], dtype=tensor.dtype, device=tensor.device
            )

        y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps)
        x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps)
        center = torch.stack([x_embed, y_embed], dim=-1).flatten(1, 2)  # b x l x 2

        if ref_size_ratios is not None:
            center = center.unsqueeze(-2).expand(
                -1, -1, len(ref_size_ratios), -1
            )  # b x l x nh x 2

            ref_size = ref_size * torch.FloatTensor(list(ref_size_ratios)).to(
                tensor_list[0]
            )
            h_embed = ref_size / size_h.unsqueeze(1)  # b x nh
            w_embed = ref_size / size_w.unsqueeze(1)  # b x nh
        else:
            h_embed = ref_size / size_h
            w_embed = ref_size / size_w

        size = torch.stack([w_embed, h_embed], dim=-1)  # b x nh x 2
        size = size.unsqueeze(1).expand_as(center)  # b x l x nh x 2

        ref_box = torch.cat([center, size], dim=-1)
        ref_windows.append(ref_box)

    ref_windows = torch.cat(ref_windows, dim=1)

    return ref_windows


def patchify(imgs, patch_size, padding=True, normalize="none"):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3]
    assert padding or imgs.shape[2] % patch_size == 0

    if padding and imgs.shape[2] % patch_size != 0:
        num_patch = math.ceil(float(imgs.shape[2]) / patch_size)
        pad_size = num_patch * patch_size - imgs.shape[2]
        imgs = F.pad(imgs, (0, pad_size, 0, pad_size))

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, (patch_size**2) * 3))

    if normalize == "global":
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (var + 1.0e-6) ** 0.5

    return x


def unpatchify(x, patch_size, orig_size=None):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * patch_size, w * patch_size))

    if orig_size is not None:
        h, w = orig_size
        imgs = imgs[..., :h, :w]

    return imgs


def clip_grad_norm(params, max_norm, with_name=False):
    if max_norm > 0:
        if with_name:
            params = list(zip(*params))[1]
        return torch.nn.utils.clip_grad_norm_(params, max_norm, foreach=True)
    else:
        if with_name:
            for name, p in params:
                if p.grad is None:
                    print(name)
                    synchronize()

            device = params[0][1].grad.device
            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(p.grad.detach(), 2.0).to(device) for _, p in params]
                ),
                2.0,
            )
        else:
            device = params[0].grad.device
            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(p.grad.detach(), 2.0).to(device) for p in params]
                ),
                2.0,
            )
        return total_norm


def normalize_period(x, offset, period):
    return (x + offset * period) / period


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def get_proposal_pos_embed(proposals, hidden_dim, scale=2 * math.pi):
    assert hidden_dim % proposals.shape[-1] == 0
    num_pos_feats = int(hidden_dim / proposals.shape[-1])
    temperature = 10000

    dim_t = torch.arange(num_pos_feats, dtype=proposals.dtype, device=proposals.device)
    dim_t = temperature ** (2 * (dim_t.div(2, rounding_mode="floor")) / num_pos_feats)
    if scale is not None:
        proposals = proposals * scale
    proposals = proposals.unbind(-1)

    pos = []
    for proposal in proposals:
        proposal = proposal[..., None] / dim_t
        proposal = torch.stack(
            (proposal[..., 0::2].sin(), proposal[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos.append(proposal)
    pos = torch.cat(pos, dim=-1)

    return pos


def get_clones(module, N):
    if N == 0:
        return []
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_norm(norm, out_channels):
    if norm == "LN":
        return nn.LayerNorm(out_channels)
    if norm == "BN":
        return nn.BatchNorm2d(out_channels)
    if norm == "GN":
        return nn.GroupNorm(32, out_channels)

    raise RuntimeError(f"norm layer should be BN | LN | GN, not {norm}")


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def filter_grads(parameters, with_name=False):
    if with_name:
        return [(name, param) for name, param in parameters if param.requires_grad]
    return [param for param in parameters if param.requires_grad]


def get_root():
    root_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.abspath(os.path.join(root_folder, ".."))

    return root_folder


def get_cache_dir(cache_dir):
    # If cache_dir path exists do not join to mmf root
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(get_root(), cache_dir)
    return cache_dir


def get_batch_size(batch_size):
    world_size = get_world_size()

    if batch_size % world_size != 0:
        raise RuntimeError(
            "Batch size {} must be divisible by number "
            "of GPUs {} used.".format(batch_size, world_size)
        )

    return batch_size // world_size


def get_absolute_path(paths):
    # String check should be first as Sequence would pass for string too
    if isinstance(paths, str):
        if not os.path.isabs(paths):
            root_dir = get_root()
            paths = os.path.join(root_dir, paths)
        return paths
    elif isinstance(paths, collections.abc.Iterable):
        return [get_absolute_path(path) for path in paths]
    else:
        raise TypeError("Paths passed to dataset should either be " "string or list")


def print_cuda_usage():
    print("Memory Allocated:", torch.cuda.memory_allocated() / (1024 * 1024))
    print("Max Memory Allocated:", torch.cuda.max_memory_allocated() / (1024 * 1024))
    print("Memory Cached:", torch.cuda.memory_cached() / (1024 * 1024))
    print("Max Memory Cached:", torch.cuda.max_memory_cached() / (1024 * 1024))


def print_model_parameters(model, writer, return_only=False):
    total_params = sum(p.numel() for p in model.parameters())
    trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if not return_only:
        writer.write(
            "Total Parameters: {}. Trained Parameters: {}".format(
                total_params, trained_params
            )
        )
    return total_params, trained_params


def get_optimizer_parameters(model):
    is_parallel = isinstance(model, nn.DataParallel) or isinstance(
        model, nn.parallel.DistributedDataParallel
    )
    has_custom = (
        hasattr(model.module, "get_optimizer_parameters")
        if is_parallel
        else hasattr(model, "get_optimizer_parameters")
    )

    if has_custom:
        parameters = (
            model.module.get_optimizer_parameters()
            if is_parallel
            else model.get_optimizer_parameters()
        )
    else:
        parameters = filter_grads(model.parameters())

    return parameters


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    return torchvision.ops.misc.interpolate(
        input, size, scale_factor, mode, align_corners
    )


def extract_grid(x, x_mask, boxes, grid_size=15, align_corners=False, roi_align=False):
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

    grid_size = grid_size * 2 if roi_align else grid_size

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
        grid = grid.view(b, -1, l, grid_size // 2, 2, grid_size // 2, 2)
        grid = grid.max(-1)[0].max(-2)[0]
    else:
        grid = grid.view(b, -1, l, grid_size, grid_size)
    grid = grid.permute(0, 2, 3, 4, 1)

    return grid


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


def concat_and_pad_masks(tensor_list):
    spatial_shape = (tensor_list[i].shape[-2:] for i in range(len(tensor_list)))
    num_tensor = sum(tensor_list[i].shape[0] for i in range(len(tensor_list)))

    shape = (num_tensor, *(max(elem) for elem in zip(*spatial_shape)))
    tensor = tensor_list[0].new_zeros(shape)
    mask = tensor_list[0].new_ones(shape).bool()

    idx = 0
    for item in tensor_list:
        b, h, w = item.shape
        tensor[idx : idx + b, :h, :w].copy_(item)
        mask[idx : idx + b, :h, :w] = False
        idx += b

    assert idx == num_tensor

    return tensor, mask


def flatten_with_shape(tensor_list, mask_list):
    """
    Params:
    :tensor_list: [(B, C, H1, W1), ..., (B, C, HN, WN)]
    :mask_list: [(B, H1, W1), ..., (B, HN, WN)]

    Return:
    :tensor_flatten: (B, L, C)
    :mask_flatten: (B, L)
    :tensor_shape: (N, 2)
    """
    assert isinstance(tensor_list, collections.abc.Sequence)
    assert len(tensor_list) > 0

    N = len(tensor_list)
    tensor_shape = torch.zeros(N, 2, dtype=torch.int64, device=tensor_list[0].device)
    tensor_flatten = []

    if mask_list is not None:
        mask_flatten = []

    for i, tensor in enumerate(tensor_list):
        new_tensor = tensor.flatten(2).permute(0, 2, 1)
        tensor_flatten.append(new_tensor)

        if mask_list is not None:
            mask = mask_list[i]
            new_mask = mask.flatten(1)
            mask_flatten.append(new_mask)
            assert tensor.shape[2] == mask.shape[1]
            assert tensor.shape[3] == mask.shape[2]
        tensor_shape[i, 0] = tensor.shape[2]
        tensor_shape[i, 1] = tensor.shape[3]

    mask_flatten = torch.cat(mask_flatten, dim=1) if mask_list is not None else None
    tensor_flatten = torch.cat(tensor_flatten, dim=1)

    return tensor_flatten, mask_flatten, tensor_shape


def view_with_shape(tensor_flatten, mask_flatten, tensor_shape):
    """
    Params:
    :tensor_flatten: (B, L, C)
    :mask_flatten: (B, L)
    :tensor_shape: (N, 2)

    Return:
    :tensor_list: [(B, C, H1, W1), ..., (B, C, HN, WN)]
    :mask_list: [(B, H1, W1), ..., (B, HN, WN)]
    """
    chunk_sizes = (tensor_shape[:, 0] * tensor_shape[:, 1]).tolist()
    N = tensor_shape.shape[0]

    if tensor_flatten is None and mask_flatten is None:
        raise ValueError("Both tensor and mask are None")
    B = tensor_flatten.shape[0] if tensor_flatten is not None else mask_flatten.shape[0]

    if tensor_flatten is not None:
        tensor_list = torch.split(tensor_flatten, chunk_sizes, dim=1)

    if mask_flatten is not None:
        mask_list = torch.split(mask_flatten, chunk_sizes, dim=1)

    tensor2d_list = [] if tensor_flatten is not None else None
    mask2d_list = [] if mask_flatten is not None else None
    for i in range(N):
        H, W = tensor_shape[i].tolist()
        if tensor_flatten is not None:
            tensor2d_list.append(
                tensor_list[i].view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            )
        if mask_flatten is not None:
            mask2d_list.append(mask_list[i].view(B, H, W))

    return tensor2d_list, mask2d_list


def split_with_shape(tensor_flatten, mask_flatten, tensor_shape):
    """
    Params:
    :tensor_flatten: (B, L, C)
    :mask_flatten: (B, L)
    :tensor_shape: (N, 2)

    Return:
    :tensor_list: [(B, H1 * W1, C), ..., (B, HN * WN, C)]
    :mask_list: [(B, H1 * W1), ..., (B, HN * WN)]
    """
    chunk_sizes = (tensor_shape[:, 0] * tensor_shape[:, 1]).tolist()

    if tensor_flatten is None and mask_flatten is None:
        raise ValueError("Both tensor and mask are None")

    if tensor_flatten is not None:
        tensor_list = torch.split(tensor_flatten, chunk_sizes, dim=1)
    else:
        tensor_list = None

    if mask_flatten is not None:
        mask_list = torch.split(mask_flatten, chunk_sizes, dim=1)
    else:
        mask_list = None

    return tensor_list, mask_list


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def data_to_tensor(data):
    data_type = type(data)

    if isinstance(data, torch.Tensor):
        return data
    elif (
        data_type.__module__ == "numpy"
        and data_type.__name__ != "str_"
        and data_type.__name__ != "string_"
    ):
        if data_type.__name__ == "ndarray" or data_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(data.dtype.str) is not None:
                return data

            return torch.as_tensor(data)
        elif data.shape == ():
            return torch.as_tensor([data.item()])

    elif isinstance(data, float):
        return torch.tensor([data], dtype=torch.float32)
    elif isinstance(data, int):
        return torch.tensor([data])
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {key: data_to_tensor(value) for key, value in data.items()}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return data_type(*(data_to_tensor(elem) for elem in data))
    elif isinstance(data, collections.abc.Sequence):
        return [data_to_tensor(elem) for elem in data]


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def contains_fsdp(model: nn.Module) -> bool:
    """
    Checks if the model contains FSDP.

    Args:
        model (nn.Module): Model to check.

    Returns:
        bool: True if the model contains FSDP, False otherwise.
    """
    return any(
        isinstance(m, torch.distributed.fsdp.FullyShardedDataParallel)
        for m in model.modules()
    )


def get_memory_stats(device: torch.device, reset_stats: bool = True) -> dict:
    """
    Computes a memory summary for the passed in device. If ``reset_stats`` is ``True``, this will
    also reset CUDA's peak memory tracking. This is useful to get data around relative use of peak
    memory (e.g. peak memory during model init, during forward, etc) and optimize memory for
    individual sections of training.

    Args:
        device (torch.device): Device to get memory summary for. Only CUDA devices are supported.
        reset_stats (bool): Whether to reset CUDA's peak memory tracking.

    Returns:
        Dict[str, float]: A dictionary containing the peak memory active, peak memory allocated,
        and peak memory reserved. This dict is useful for logging memory stats.

    Raises:
        ValueError: If the passed-in device is not CUDA.
    """
    if device.type != "cuda":
        raise ValueError(
            f"Logging memory stats is only supported on CUDA devices, got {device}"
        )

    peak_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / (
        1024**3
    )
    peak_mem_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
    peak_mem_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)

    if reset_stats:
        torch.cuda.reset_peak_memory_stats(device)

    memory_stats = {
        "peak_memory_active": peak_memory_active,
        "peak_memory_alloc": peak_mem_alloc,
        "peak_memory_reserved": peak_mem_reserved,
    }
    return memory_stats


def cleanup_before_training() -> None:
    """
    Call gc collect, empty CUDA cache, and reset peak memory stats.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def set_determinism(seed) -> None:
    """
    Set Python, PyTorch, CUDA seeds and cudnn settings for reproducibility
    """
    if seed > 0:
        # CPU and GPU determinism
        torch.manual_seed(seed)
        # set deterministic cudnn algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # set Python seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.use_deterministic_algorithms(True)
        # env var for deterministic CuBLAS
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # ensure we turn off deterministic cudnn algorithms
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def module_filter_fn(mod: nn.Module, fqn: str, filter_fqns: list[str]) -> bool:
    """
    Filter function to determine which modules should be converted.
    For both Float8 and MXFP8, we only convert Linear modules
    with dimensions divisible by 16 and not matching any filtered FQNs.
    """
    if not isinstance(mod, nn.Linear):
        return False

    # All dims must be divisible by 16 due to float8 tensorcore hardware requirements.
    dims_multiples_of_16 = (
        mod.weight.shape[0] % 16 == 0 and mod.weight.shape[1] % 16 == 0
    )

    # If the fqn matches any filtered fqn, then we should not convert this module.
    is_filtered_fqn = any(filter_fqn in fqn for filter_fqn in filter_fqns)

    return dims_multiples_of_16 and is_filtered_fqn


# used to avoid stragglers in garbage collection
class GarbageCollection:
    def __init__(self, gc_freq: int = 1000, debug: bool = False):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        self.debug = debug
        gc.disable()
        self.collect("Initial GC collection.")
        if debug:
            from torch.utils.viz._cycles import warn_tensor_cycles

            if torch.distributed.get_rank() == 0:
                warn_tensor_cycles()

    def run(self, step_count: int):
        if self.debug:
            self.collect(
                "Force GC to perform collection to obtain debug information.",
                generation=2,
            )
            gc.collect()
        elif step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Peforming periodical GC collection.")

    @staticmethod
    def collect(reason: str, generation: int = 1):
        begin = time.monotonic()
        gc.collect(generation)
        print("[GC] %s %.2f seconds.", reason, time.monotonic() - begin)
