import math
import os
import copy
from collections import OrderedDict, abc
from functools import partial

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 10):
    from torchvision._internally_replaced_utils import load_state_dict_from_url
else:
    from torchvision.models.utils import load_state_dict_from_url

from .utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    get_rel_pos_bias,
    window_partition,
    window_unpartition,
    RelativePositionBias,
)
from .helper import DropPath, Mlp, trunc_normal_, LayerNorm, Upscale2d
from e2edet.module.transformer import build_position_encoding
from e2edet.utils.distributed import synchronize, is_master


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

model_urls = {
    "mae_base_patch16": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
    "mae_large_patch16": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
    "mae_huge_patch16": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
    "beitv2_base_patch16_1k": "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k.pth",
    "beitv2_base_patch16": "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth",
    "beitv2_large_patch16_1k": "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k.pth",
    "beitv2_large_patch16": "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth",
    "vit_base_patch16_224": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
    "moco_base_patch16": "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
    "pmae_base_patch16": "https://dl.fbaipublicfiles.com/mae/share/default.pth",
    "deit_base_patch16": "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
    "deitv3_base_patch16": "https://dl.fbaipublicfiles.com/deit/deit_3_base_384_1k.pth",
    "deitv3_base_patch16_21k": "https://dl.fbaipublicfiles.com/deit/deit_3_base_384_21k.pth",
    "deitv3_base_patch16_21k_224": "https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth",
    "deitv3_large_patch16_21k": "https://dl.fbaipublicfiles.com/deit/deit_3_large_384_21k.pth",
    "pmae_base_patch16_224000": "https://dl.fbaipublicfiles.com/mae/share/max_update_224000.pth",
    "mae_max_update_112000": "https://dl.fbaipublicfiles.com/mae/share/mae_only_max_update_112000.pth",
    "mae_max_update_224000": "https://dl.fbaipublicfiles.com/mae/share/mae_only_max_update_224000.pth",
}


def get_norm(norm, out_channels):
    if norm == "LN":
        return LayerNorm(out_channels, data_format="channels_first")
    if norm == "BN":
        return nn.BatchNorm2d(out_channels)
    if norm == "GN":
        return nn.GroupNorm(32, out_channels)

    raise RuntimeError(f"norm layer should be BN | LN | GN, not {norm}")


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        # self.attn = attn.view(B, self.num_heads, H * W, H * W)

        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)

        return x


class AttentionForBeit(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
        use_rel_pos_bias=True,
        pretrain_input_size=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

        self.use_rel_pos_bias = use_rel_pos_bias
        if self.use_rel_pos_bias:
            window_size = pretrain_input_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(
                torch.meshgrid([coords_h, coords_w], indexing="ij")
            )  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
            self.register_buffer("rel_pos_idx", relative_coords.sum(-1))

            self.window_size = pretrain_input_size
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, relative_position_bias=None):
        B, H, W, _ = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v = (
        #     qkv[0],
        #     qkv[1],
        #     qkv[2],
        # )  # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.rel_pos_idx.view(-1)
            ]
            relative_position_bias = relative_position_bias.view(
                self.window_size[0],
                self.window_size[1],
                self.window_size[0],
                self.window_size[1],
                -1,
            )

            attn = attn.view(B, self.num_heads, H * W, H * W) + get_rel_pos_bias(
                relative_position_bias, (H, W)
            )
            attn = attn.view(B * self.num_heads, H * W, -1)
        elif relative_position_bias is not None:
            attn = attn.view(B, self.num_heads, H * W, H * W) + get_rel_pos_bias(
                relative_position_bias, (H, W)
            )
            attn = attn.view(B * self.num_heads, H * W, -1)

        attn = attn.softmax(dim=-1)
        # self.attn = attn.view(B, self.num_heads, H * W, H * W)
        attn = self.attn_drop(attn)

        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer
        )

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BlockForDeitv3(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
        init_values=1e-4,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


class BlockForBeit(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
        use_rel_pos_bias=True,
        shared_rel_pos_bias=False,
        pretrain_input_size=None,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = AttentionForBeit(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rel_pos_bias=use_rel_pos_bias if not shared_rel_pos_bias else False,
            pretrain_input_size=pretrain_input_size,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.window_size = window_size

    def forward(self, x, relative_position_bias=None):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        if self.gamma_1 is None:
            x = self.attn(x, relative_position_bias)
        else:
            x = self.gamma_1 * self.attn(x, relative_position_bias)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)

        if self.gamma_2 is None:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


class VisionTransformerDet(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=[],
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
        enc_norm=True,
        checkpoint_layer=-1,
        freeze_backbone=False,
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size
            )
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        if enc_norm is True:
            self.norm = norm_layer(embed_dim)
        elif enc_norm is False:
            self.last_norm = norm_layer(embed_dim)
        assert enc_norm in (True, False, "none")

        self.enc_norm = enc_norm

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.patch_size = patch_size
        self.depth = depth
        self.checkpoint_layer = checkpoint_layer
        self.apply(self._init_weights)

        if freeze_backbone:
            print("Freezing backbone...")
            for params in self.parameters():
                params.requires_grad_(False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def output_shape(self):
        return {
            name: dict(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        for i, blk in enumerate(self.blocks):
            if i < self.checkpoint_layer:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.enc_norm is True:
            x = self.norm(x)
        elif self.enc_norm is False:
            x = self.last_norm(x)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}

        return outputs


class VisionTransformerDetForBeit(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.1,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        window_size=0,
        window_block_indexes=[],
        use_rel_pos=False,
        rel_pos_zero_init=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True,
        out_feature="last_feat",
        checkpoint_layer=-1,
        shared_rel_pos_bias=False,
        **__,
    ):
        super().__init__()

        if shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=(
                    pretrain_img_size // patch_size,
                    pretrain_img_size // patch_size,
                ),
                num_heads=num_heads,
            )
        else:
            self.rel_pos_bias = None

        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos_emb:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size
            )
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                BlockForBeit(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    use_rel_pos=use_rel_pos,
                    rel_pos_zero_init=rel_pos_zero_init,
                    window_size=window_size if i in window_block_indexes else 0,
                    input_size=(img_size // patch_size, img_size // patch_size),
                    use_rel_pos_bias=use_rel_pos_bias,
                    shared_rel_pos_bias=shared_rel_pos_bias,
                    pretrain_input_size=(
                        pretrain_img_size // patch_size,
                        pretrain_img_size // patch_size,
                    ),
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        self.patch_size = patch_size
        self.depth = depth
        self.checkpoint_layer = checkpoint_layer

        # if self.pos_embed is not None:
        #     trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def output_shape(self):
        return {
            name: dict(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    def get_num_layers(self):
        return len(self.blocks)

    def forward(self, x):
        if self.rel_pos_bias is not None:
            relative_position_bias = self.rel_pos_bias()
        else:
            relative_position_bias = None

        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i < self.checkpoint_layer:
                x = checkpoint.checkpoint(blk, x, relative_position_bias)
            else:
                x = blk(x, relative_position_bias)

        x = self.norm(x)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}

        return outputs


class VisionTransformerDetForDeit(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=[],
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        pretrain_use_distill_token=True,
        out_feature="last_feat",
        enc_norm=True,
        checkpoint_layer=-1,
        freeze_backbone=False,
        use_block_for_deit=False,
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.pretrain_use_distill_token = pretrain_use_distill_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size
            )

            num_tokens = 0
            if pretrain_use_cls_token:
                num_tokens += 1

            if pretrain_use_distill_token:
                num_tokens += 1

            num_positions = num_patches + num_tokens
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        block_cls = Block if not use_block_for_deit else BlockForDeitv3
        for i in range(depth):
            block = block_cls(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        if enc_norm is True:
            self.norm = norm_layer(embed_dim)
        elif enc_norm is False:
            self.last_norm = norm_layer(embed_dim)
        assert enc_norm in (True, False, "none")

        self.enc_norm = enc_norm

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.patch_size = patch_size
        self.depth = depth
        self.checkpoint_layer = checkpoint_layer
        self.apply(self._init_weights)

        if freeze_backbone:
            print("Freezing backbone...")
            for params in self.parameters():
                params.requires_grad_(False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def output_shape(self):
        return {
            name: dict(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            if self.pretrain_use_distill_token:
                x = x + get_abs_pos(
                    self.pos_embed[:, 1:],
                    self.pretrain_use_cls_token,
                    (x.shape[1], x.shape[2]),
                )
            else:
                x = x + get_abs_pos(
                    self.pos_embed,
                    self.pretrain_use_cls_token,
                    (x.shape[1], x.shape[2]),
                )

        for i, blk in enumerate(self.blocks):
            if i < self.checkpoint_layer:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.enc_norm is True:
            x = self.norm(x)
        elif self.enc_norm is False:
            x = self.last_norm(x)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}

        return outputs


class ViTUp(nn.Module):
    def __init__(
        self,
        net,
        in_features,
        embed_dim,
        scale_factors,
        hidden_dim=256,
        position_encoding=None,
        ref_size=4,
        norm="GN",
        upscale_mode="conv",
    ):
        super(ViTUp, self).__init__()
        assert len(in_features) == len(scale_factors)

        self.scale_factors = scale_factors
        input_shapes = net.output_shape()
        self.net = net
        self._out_features = in_features
        self._out_feature_channels = {
            f: input_shapes[f]["channels"] for f in in_features
        }
        self._out_feature_strides = {
            f: int(input_shapes[f]["stride"] / scale / 2)
            for f, scale in zip(in_features, scale_factors)
        }
        print(self._out_feature_channels, self._out_feature_strides)

        if upscale_mode == "conv":
            self.upscale_layer = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, hidden_dim, kernel_size=2, stride=2),
                get_norm(norm, hidden_dim),
            )
        elif upscale_mode == "identity":
            self.upscale_layer = nn.Sequential(
                Upscale2d(scale_factor=2),
                nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, padding=2),
                get_norm(norm, hidden_dim),
            )

        self.ref_size = ref_size
        self.position_encoding = None
        if position_encoding is not None:
            self.position_encoding = build_position_encoding(
                position_encoding, hidden_dim
            )

    @torch.jit.ignore
    def shard_modules(self):
        return {"net.blocks"}

    def forward(self, x, mask):
        features = self.net(x, mask)

        outputs = {}
        for i, scale in enumerate(self.scale_factors):
            if scale != 1:
                f = F.interpolate(
                    features[self._out_features[i]],
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                f = features[self._out_features[i]]
            f = self.upscale_layer(f)

            if mask is not None:
                f_mask = F.interpolate(mask[None].float(), size=f.shape[-2:]).bool()[0]
            else:
                f_mask = None

            if self.position_encoding is not None:
                f_pos = self.position_encoding(f, f_mask, self.ref_size)
            else:
                f_pos = None
            outputs[self._out_features[i]] = (f, f_mask, f_pos)

        return outputs


class ViTUp1(nn.Module):
    def __init__(
        self,
        net,
        in_features,
        embed_dim,
        scale_factors,
        out_features,
        hidden_dim=256,
        position_encoding=None,
        ref_size=4,
        norm="GN",
        prenorm=False,
    ):
        super(ViTUp1, self).__init__()

        self.scale_factors = scale_factors
        self.in_feature = in_features

        input_shapes = net.output_shape()
        self.net = net
        self._out_features = out_features
        self._out_feature_channels = {f: hidden_dim for f in out_features}
        self._out_feature_strides = {
            f: int(input_shapes[f]["stride"] / scale)
            for f, scale in zip(out_features, scale_factors)
        }
        print(self._out_feature_channels, self._out_feature_strides)

        self.ref_size = ref_size
        self.position_encoding = None
        if position_encoding is not None:
            self.position_encoding = build_position_encoding(
                position_encoding, hidden_dim
            )

        if prenorm:
            self.conv = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
                get_norm(norm, hidden_dim),
            )

    @torch.jit.ignore
    def shard_modules(self):
        return {"net.blocks"}

    def forward(self, x, mask):
        feature = self.net(x)
        feature = self.conv(feature[self.in_feature])

        outputs = {}
        for i, scale in enumerate(self.scale_factors):
            if scale != 1:
                f = F.interpolate(
                    feature, scale_factor=scale, mode="bilinear", align_corners=False
                )
            else:
                f = feature

            if mask is not None:
                f_mask = F.interpolate(mask[None].float(), size=f.shape[-2:]).bool()[0]
            else:
                f_mask = None

            if self.position_encoding is not None:
                f_pos = self.position_encoding(f, f_mask, self.ref_size)
            else:
                f_pos = None
            outputs[self._out_features[i]] = (f, f_mask, f_pos)

        return outputs


class ViTUpFirstUp(nn.Module):
    """[summary]
    input: single scale
    Upsample first, then downsample for four strides.
    """

    def __init__(
        self,
        net,
        in_features,
        embed_dim,
        scale_factors,
        mode,
        out_features,
        upscale,
        hidden_dim=256,
        position_encoding=None,
        ref_size=4,
        norm="GN",
    ):
        super(ViTUpFirstUp, self).__init__()
        assert len(out_features) == len(scale_factors)

        self.scale_factors = scale_factors
        self.upscale = upscale
        self.in_feature = in_features

        input_shapes = net.output_shape()
        self.net = net
        self._out_features = out_features
        self._out_feature_channels = {f: hidden_dim for f in out_features}
        self._out_feature_strides = {
            f: int(input_shapes[self.in_feature]["stride"] / upscale / scale)
            for f, scale in zip(out_features, scale_factors)
        }
        print(
            "resize",
            self._out_feature_channels,
            self._out_feature_strides,
            input_shapes,
        )

        self.ref_size = ref_size
        self.position_encoding = None
        if position_encoding is not None:
            self.position_encoding = build_position_encoding(
                position_encoding, hidden_dim
            )

        assert upscale == 2
        self.upscale_layer = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, hidden_dim, kernel_size=2, stride=2),
            get_norm(norm, hidden_dim),
        )
        assert mode in ["max", "conv", "avg"]
        # assert mode != "max" or out_dim == embed_dim

        for i, scale in enumerate(scale_factors):
            if scale == 1 / 8.0:
                if mode == "max":
                    layer = nn.MaxPool2d(kernel_size=8, stride=8)
                elif mode == "avg":
                    layer = nn.AvgPool2d(kernel_size=8, stride=8)
                else:
                    raise ValueError("only max|avg are supported")
            elif scale == 1 / 4.0:
                if mode == "max":
                    layer = nn.MaxPool2d(kernel_size=4, stride=4)
                elif mode == "avg":
                    layer = nn.AvgPool2d(kernel_size=4, stride=4)
                else:
                    raise ValueError("only max|avg are supported")
            elif scale == 1 / 2.0:
                if mode == "max":
                    layer = nn.MaxPool2d(kernel_size=2, stride=2)
                elif mode == "avg":
                    layer = nn.AvgPool2d(kernel_size=2, stride=2)
                else:
                    raise ValueError("only max|avg are supported")
            elif scale == 1.0:
                layer = nn.Identity()
            else:
                raise NotImplementedError

            self.add_module(f"stage_{i}", layer)

    @torch.jit.ignore
    def shard_modules(self):
        return {"net.blocks"}

    def forward(self, x, mask):
        features = self.net(x)

        feature = self.upscale_layer(features[self.in_feature])

        outputs = {}
        for i in range(len(self.scale_factors)):
            f = getattr(self, f"stage_{i}")(feature)

            if mask is not None:
                f_mask = F.interpolate(mask[None].float(), size=f.shape[-2:]).bool()[0]
            else:
                f_mask = None

            if self.position_encoding is not None:
                f_pos = self.position_encoding(f, f_mask, self.ref_size)
            else:
                f_pos = None
            outputs[self._out_features[i]] = (f, f_mask, f_pos)

        return outputs


class ViTUpFirstUp1(nn.Module):
    """[summary]
    input: single scale
    Upsample first, then downsample for four strides.
    """

    def __init__(
        self,
        net,
        in_features,
        embed_dim,
        scale_factors,
        mode,
        out_features,
        upscale,
        hidden_dim=256,
        mask_dim=128,
        position_encoding=None,
        ref_size=4,
        norm="GN",
        use_feat=True,
    ):
        super(ViTUpFirstUp1, self).__init__()
        assert len(out_features) == len(scale_factors)
        assert (
            "feat" not in out_features
        ), "feat is reserved for high-resolution features"

        self.scale_factors = scale_factors
        self.upscale = upscale
        self.in_feature = in_features

        input_shapes = net.output_shape()
        self.net = net
        self._out_features = out_features
        self._out_feature_channels = {f: hidden_dim for f in out_features}
        self._out_feature_strides = {
            f: int(input_shapes[self.in_feature]["stride"] / upscale / scale)
            for f, scale in zip(out_features, scale_factors)
        }
        print(
            "resize",
            self._out_feature_channels,
            self._out_feature_strides,
            input_shapes,
        )

        self.ref_size = ref_size
        self.use_feat = use_feat
        self.position_encoding = None
        if position_encoding is not None:
            self.position_encoding = build_position_encoding(
                position_encoding, hidden_dim
            )

        if upscale == 2:
            self.upscale_layer = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim, hidden_dim, kernel_size=2, stride=2
                        ),
                        get_norm(norm, hidden_dim),
                    ),
                ]
            )
        elif upscale == 1:
            self.upscale_layer = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
                        get_norm(norm, hidden_dim),
                    ),
                ]
            )
        elif upscale == 4:
            self.upscale_layer = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            embed_dim, embed_dim // 2, kernel_size=2, stride=2
                        ),
                        LayerNorm(embed_dim // 2, data_format="channels_first"),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            embed_dim // 2, hidden_dim, kernel_size=2, stride=2
                        ),
                        get_norm(norm, hidden_dim),
                    ),
                ]
            )
        else:
            raise ValueError("Only 2 or 4 upscale are allowed!")
        if use_feat:
            self.upscale_layer.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim // 2, kernel_size=2, stride=2
                    ),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        embed_dim // 2, mask_dim, kernel_size=2, stride=2
                    ),
                )
            )
        assert mode in ["max", "conv", "avg"]
        # assert mode != "max" or out_dim == embed_dim

        for i, scale in enumerate(scale_factors):
            if scale == 1 / 8.0:
                if mode == "max":
                    layer = nn.MaxPool2d(kernel_size=8, stride=8)
                elif mode == "avg":
                    layer = nn.AvgPool2d(kernel_size=8, stride=8)
                else:
                    raise ValueError("only max|avg are supported")
            elif scale == 1 / 4.0:
                if mode == "max":
                    layer = nn.MaxPool2d(kernel_size=4, stride=4)
                elif mode == "avg":
                    layer = nn.AvgPool2d(kernel_size=4, stride=4)
                else:
                    raise ValueError("only max|avg are supported")
            elif scale == 1 / 2.0:
                if mode == "max":
                    layer = nn.MaxPool2d(kernel_size=2, stride=2)
                elif mode == "avg":
                    layer = nn.AvgPool2d(kernel_size=2, stride=2)
                else:
                    raise ValueError("only max|avg are supported")
            elif scale == 1.0:
                layer = nn.Identity()
            else:
                raise NotImplementedError

            self.add_module(f"stage_{i}", layer)

    @torch.jit.ignore
    def shard_modules(self):
        return {"net.blocks"}

    def forward(self, x, mask):
        features = self.net(x)

        feature = self.upscale_layer[0](features[self.in_feature])

        outputs = {}
        for i in range(len(self.scale_factors)):
            f = getattr(self, f"stage_{i}")(feature)

            if mask is not None:
                f_mask = F.interpolate(mask[None].float(), size=f.shape[-2:]).bool()[0]
            else:
                f_mask = None

            if self.position_encoding is not None:
                f_pos = self.position_encoding(f, f_mask, self.ref_size)
            else:
                f_pos = None

            outputs[self._out_features[i]] = (f, f_mask, f_pos)

        if self.use_feat:
            feat = self.upscale_layer[1](features[self.in_feature])
            if mask is not None:
                feat_mask = F.interpolate(
                    mask[None].float(), size=feat.shape[-2:]
                ).bool()[0]
            else:
                feat_mask = None
            outputs["feat"] = (feat, feat_mask)
        else:
            outputs["feat"] = (None, None)

        return outputs


class SimpleFPN(nn.Module):
    def __init__(
        self,
        net,
        in_features,
        embed_dim,
        scale_factors,
        mode,
        out_features,
        upscale,
        hidden_dim=256,
        position_encoding=None,
        ref_size=4,
        norm="GN",
        **__,
    ):
        super(SimpleFPN, self).__init__()
        assert len(out_features) == len(scale_factors)

        self.scale_factors = scale_factors
        self.upscale = upscale
        self.in_feature = in_features

        input_shapes = net.output_shape()
        self.net = net
        self._out_features = out_features
        self._out_feature_channels = {f: hidden_dim for f in out_features}
        self._out_feature_strides = {
            f: int(input_shapes[self.in_feature]["stride"] / upscale / scale)
            for f, scale in zip(out_features, scale_factors)
        }
        print(
            "resize",
            self._out_feature_channels,
            self._out_feature_strides,
            input_shapes,
        )

        self.ref_size = ref_size
        self.position_encoding = None
        if position_encoding is not None:
            self.position_encoding = build_position_encoding(
                position_encoding, hidden_dim
            )

        # assert upscale == 2
        assert mode in ["max", "conv", "avg"]
        # assert mode != "max" or out_dim == embed_dim

        for i, scale in enumerate(scale_factors):
            out_dim = embed_dim
            if scale == 0.25:
                if mode == "max":
                    layers = [nn.MaxPool2d(kernel_size=4, stride=4)]
                elif mode == "avg":
                    layers = [nn.AvgPool2d(kernel_size=4, stride=4)]
            elif scale == 0.5:
                if mode == "max":
                    layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
                elif mode == "avg":
                    layers = [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif scale == 1.0:
                layers = []
            elif scale == 2.0:
                layers = [
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim // 2, kernel_size=2, stride=2
                    )
                ]
                out_dim = embed_dim // 2
            elif scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim // 2, kernel_size=2, stride=2
                    ),
                    get_norm("LN", embed_dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2
                    ),
                ]
                out_dim = embed_dim // 4
            else:
                raise NotImplementedError

            layers.extend(
                [
                    nn.Conv2d(out_dim, hidden_dim, kernel_size=1, bias=False),
                    get_norm("LN", hidden_dim),
                    nn.Conv2d(
                        hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False
                    ),
                    get_norm(norm, hidden_dim),
                ]
            )
            layers = nn.Sequential(*layers)

            self.add_module(f"stage_{i}", layers)

    @torch.jit.ignore
    def shard_modules(self):
        return {"net.blocks"}

    def forward(self, x, mask):
        feature = self.net(x)[self.in_feature]

        outputs = {}
        for i in range(len(self.scale_factors)):
            f = getattr(self, f"stage_{i}")(feature)

            if mask is not None:
                f_mask = F.interpolate(mask[None].float(), size=f.shape[-2:]).bool()[0]
            else:
                f_mask = None

            if self.position_encoding is not None:
                f_pos = self.position_encoding(f, f_mask, self.ref_size)
            else:
                f_pos = None
            outputs[self._out_features[i]] = (f, f_mask, f_pos)

        return outputs


class SimpleFPN1(nn.Module):
    def __init__(
        self,
        net,
        in_features,
        embed_dim,
        scale_factors,
        mode,
        out_features,
        upscale,
        hidden_dim=256,
        mask_dim=128,
        position_encoding=None,
        ref_size=4,
        norm="GN",
        **__,
    ):
        super(SimpleFPN1, self).__init__()
        assert len(out_features) == len(scale_factors)

        self.scale_factors = scale_factors
        self.upscale = upscale
        self.in_feature = in_features

        input_shapes = net.output_shape()
        self.net = net
        self._out_features = out_features
        self._out_feature_channels = {f: hidden_dim for f in out_features}
        self._out_feature_strides = {
            f: int(input_shapes[self.in_feature]["stride"] / upscale / scale)
            for f, scale in zip(out_features, scale_factors)
        }
        print(
            "resize",
            self._out_feature_channels,
            self._out_feature_strides,
            input_shapes,
        )

        self.ref_size = ref_size
        self.position_encoding = None
        if position_encoding is not None:
            self.position_encoding = build_position_encoding(
                position_encoding, hidden_dim
            )

        # assert upscale == 2
        assert mode in ["max", "conv", "avg"]
        # assert mode != "max" or out_dim == embed_dim

        for i, scale in enumerate(scale_factors):
            out_dim = embed_dim
            if scale == 0.25:
                if mode == "max":
                    layers = [nn.MaxPool2d(kernel_size=4, stride=4)]
                elif mode == "avg":
                    layers = [nn.AvgPool2d(kernel_size=4, stride=4)]
            elif scale == 0.5:
                if mode == "max":
                    layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
                elif mode == "avg":
                    layers = [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif scale == 1.0:
                layers = []
            elif scale == 2.0:
                layers = [
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim // 2, kernel_size=2, stride=2
                    )
                ]
                out_dim = embed_dim // 2
            else:
                raise NotImplementedError

            layers.extend(
                [
                    nn.Conv2d(out_dim, hidden_dim, kernel_size=1, bias=False),
                    get_norm("LN", hidden_dim),
                    nn.Conv2d(
                        hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False
                    ),
                    get_norm(norm, hidden_dim),
                ]
            )
            layers = nn.Sequential(*layers)
            self.add_module(f"stage_{i}", layers)

        self.upscale_layer = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            get_norm("LN", embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.Conv2d(embed_dim // 4, mask_dim, kernel_size=1, bias=False),
            get_norm("LN", mask_dim),
            nn.Conv2d(mask_dim, mask_dim, kernel_size=3, padding=1, bias=False),
            get_norm(norm, mask_dim),
        )

    @torch.jit.ignore
    def shard_modules(self):
        return {"net.blocks"}

    def forward(self, x, mask):
        feature = self.net(x)[self.in_feature]

        outputs = {}
        for i in range(len(self.scale_factors)):
            f = getattr(self, f"stage_{i}")(feature)

            if mask is not None:
                f_mask = F.interpolate(mask[None].float(), size=f.shape[-2:]).bool()[0]
            else:
                f_mask = None

            if self.position_encoding is not None:
                f_pos = self.position_encoding(f, f_mask, self.ref_size)
            else:
                f_pos = None
            outputs[self._out_features[i]] = (f, f_mask, f_pos)

        feat = self.upscale_layer(feature)
        if mask is not None:
            feat_mask = F.interpolate(mask[None].float(), size=feat.shape[-2:]).bool()[
                0
            ]
        else:
            feat_mask = None
        outputs["feat"] = (feat, feat_mask)

        return outputs


def _load_state_dict(arch, pretrained, pretrained_path, model_data_dir):
    if arch.startswith("mvit"):
        assert pretrained is False, "mvit backbone should not be pretrained"

    state_dict = None
    if pretrained:
        if pretrained_path is None:
            # Prevent nodes from caching state_dicts at the same time
            if is_master():
                load_state_dict_from_url(
                    model_urls[arch], map_location=torch.device("cpu")
                )
            synchronize()
            state_dict = load_state_dict_from_url(
                model_urls[arch], map_location=torch.device("cpu")
            )
        else:
            if not os.path.isabs(pretrained_path):
                pretrained_path = os.path.join(model_data_dir, pretrained_path)

            if not os.path.exists(pretrained_path):
                raise ValueError(
                    "pretrained_path ({}) doesn't exist".format(pretrained_path)
                )
            state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))

    if state_dict is not None:
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            new_state_dict = OrderedDict()
            for kk, vv in state_dict.items():
                new_state_dict[kk.replace("module.base_encoder.", "")] = vv
            state_dict = new_state_dict

        if "huge" in arch:
            for kk, vv in state_dict.items():
                if "patch_embed.proj.weight" in kk:
                    state_dict[kk] = F.interpolate(
                        vv, size=(16, 16), mode="bicubic", align_corners=False
                    )

                if "pos_embed" in kk:
                    pos_embed = torch.cat(
                        [
                            vv[:, :1, :],
                            F.interpolate(
                                vv[:, 1:, :].reshape(1, 16, 16, -1).permute(0, 3, 1, 2),
                                size=(14, 14),
                                mode="bicubic",
                                align_corners=False,
                            )
                            .flatten(2)
                            .transpose(1, 2),
                        ],
                        dim=1,
                    )
                    state_dict[kk] = pos_embed

    # if pretrained_mode is not None:
    #     state_dict = _gather_state_dict(state_dict, mode=pretrained_mode)

    return state_dict


def _build_vit_det(
    arch, patch_size, embed_dim, depth, num_heads, state_dict=None, **kwargs
):
    if "beit" in arch:
        model = VisionTransformerDetForBeit(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    elif "deit" in arch:
        model = VisionTransformerDetForDeit(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    else:
        model = VisionTransformerDet(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
    if state_dict is not None:
        named_tuple = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained {arch}:", named_tuple)
        del state_dict

    return model


def build_vit_det(config):
    ms_type = config["ms_type"]
    ms_params = copy.deepcopy(config["ms_params"])

    arch = config["arch"]
    params = copy.deepcopy(config["params"])

    with omegaconf.open_dict(params):
        pretrained = params.pop("pretrained", True)
        pretrained_path = params.pop("pretrained_path", None)
        model_data_dir = params.pop("model_data_dir", None)
        window_block_indexes = params.get("window_block_indexes", None)

        if arch in (
            "vit_base_patch16_224",
            "vit_base_patch16_224_in21k",
            "mae_base_patch16",
            "moco_base_patch16",
            "pmae_base_patch16",
            "mae_max_update_112000",
            "pmae_base_patch16_224000",
            "mae_max_update_224000",
            "beitv2_base_patch16",
        ):
            patch_size = 16
            embed_dim = 768
            depth = 12
            num_heads = 12
        elif arch in ("beitv2_base_patch16_1k",):
            patch_size = 16
            embed_dim = 768
            depth = 12
            num_heads = 12
            params["shared_rel_pos_bias"] = True
        elif arch in (
            "vit_base_patch16_384",
            "deit_base_patch16",
        ):
            patch_size = 16
            embed_dim = 768
            depth = 12
            num_heads = 12
            params["pretrain_img_size"] = 384
        elif arch in (
            "deitv3_base_patch16_21k",
            "deitv3_base_patch16",
        ):
            patch_size = 16
            embed_dim = 768
            depth = 12
            num_heads = 12
            params["pretrain_img_size"] = 384
            params["pretrain_use_distill_token"] = False
            params["pretrain_use_cls_token"] = False
            params["use_block_for_deit"] = True
        elif arch in (
            "vit_large_patch16_224",
            "vit_large_patch16_224_in21k",
            "mae_large_patch16",
            "beitv2_large_patch16",
        ):
            patch_size = 16
            embed_dim = 1024
            depth = 24
            num_heads = 16
        elif arch in ("beitv2_large_patch16_1k",):
            patch_size = 16
            embed_dim = 1024
            depth = 24
            num_heads = 16
            params["shared_rel_pos_bias"] = True
        elif arch in ("vit_large_patch16_384",):
            patch_size = 16
            embed_dim = 1024
            depth = 24
            num_heads = 16
            params["pretrain_img_size"] = 384
        elif arch in ("deitv3_large_patch16_21k",):
            patch_size = 16
            embed_dim = 1024
            depth = 24
            num_heads = 16
            params["pretrain_img_size"] = 384
            params["pretrain_use_distill_token"] = False
            params["pretrain_use_cls_token"] = False
            params["use_block_for_deit"] = True
        elif arch in (
            "vit_huge_patch16_224_in21k",
            "mae_huge_patch16",
        ):
            patch_size = 16
            embed_dim = 1280
            depth = 32
            num_heads = 16
        else:
            if pretrained:
                raise RuntimeError("pretrained is not supported with your config!")

        if window_block_indexes == "all":
            params["window_block_indexes"] = list(range(depth))
        elif isinstance(window_block_indexes, int):
            params["window_block_indexes"] = list(range(window_block_indexes))
        elif isinstance(window_block_indexes, abc.Sequence):
            params["window_block_indexes"] = window_block_indexes

    if ms_type == "vit_up":
        ms_cls = ViTUp
    elif ms_type == "vit_up1":
        ms_cls = ViTUp1
    elif ms_type == "vit_up_fist_up":
        ms_cls = ViTUpFirstUp
    elif ms_type == "vit_up_fist_up1":
        ms_cls = ViTUpFirstUp1
    elif ms_type == "SimpleFPN":
        ms_cls = SimpleFPN
    elif ms_type == "SimpleFPN1":
        ms_cls = SimpleFPN1
    else:
        raise ValueError("Wrong ms_type:", ms_type)

    state_dict = _load_state_dict(arch, pretrained, pretrained_path, model_data_dir)

    # if pretrained_mode == "backbone":
    #     net = _build_vit_det(arch, patch_size, embed_dim, depth, num_heads, **params)
    #     model = ms_cls(net, params["out_feature"], embed_dim, **ms_params)
    #     named_tuple = model.load_state_dict(state_dict, strict=False)
    #     print(f"Loaded pretrained backbone:", named_tuple)
    # else:
    net = _build_vit_det(
        arch,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        state_dict=state_dict,
        **params,
    )

    model = ms_cls(net, params["out_feature"], embed_dim, **ms_params)

    return model
