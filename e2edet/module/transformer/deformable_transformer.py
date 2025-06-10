import warnings
import math

import torch
import torch.nn as nn

from ..attention.deformable_attention import DeformableAttention
from e2edet.utils.general import (
    flatten_with_shape,
    inverse_sigmoid,
    get_clones,
    get_activation_fn,
    get_proposal_pos_embed,
)
from e2edet.module.head import Detector
from e2edet.module.transformer import register_transformer


@register_transformer("deformable_transformer")
class DeformableTransformer(nn.Module):
    def __init__(
        self,
        enc_dim=256,
        dec_dim=256,
        nhead=8,
        nlevel=4,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward_ratio=4.0,
        dropout=0.1,
        activation="relu",
        num_queries=300,
        num_classes=91,
        ref_size=4,
        loss_mode="focal",
        prenorm=False,
        **kwargs,
    ):
        super().__init__()

        if len(list(kwargs.keys())) > 0:
            warnings.warn(
                f"Arguments {list(kwargs.keys())} are unused in {self.__class__.__name__}"
            )

        encoder_layer = DeformableTransformerEncoderLayer(
            enc_dim,
            nhead,
            nlevel,
            enc_dim * dim_feedforward_ratio,
            dropout,
            activation,
            prenorm=prenorm,
        )
        encoder_detector = Detector(enc_dim, 1, loss_mode=loss_mode)

        self.encoder = DeformableTransformerEncoder(
            enc_dim,
            dec_dim,
            encoder_layer,
            encoder_detector,
            enc_layers,
            num_queries,
            loss_mode,
            prenorm=prenorm,
        )

        decoder_layer = DeformableTransformerDecoderLayer(
            dec_dim,
            enc_dim,
            nhead,
            nlevel,
            dec_dim * dim_feedforward_ratio,
            dropout,
            activation,
            prenorm=prenorm,
        )
        decoder_detector = Detector(dec_dim, num_classes, loss_mode=loss_mode)

        self.decoder = DeformableTransformerDecoder(
            dec_dim, decoder_layer, decoder_detector, dec_layers, prenorm=prenorm
        )
        self.ref_size = ref_size
        self.loss_mode = loss_mode

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, DeformableAttention):
                m._reset_parameters()

        if self.loss_mode in ("bce", "focal"):
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.encoder.detector.class_embed.bias, bias_value)
            nn.init.constant_(self.decoder.detector.class_embed.bias, bias_value)
        else:
            nn.init.constant_(self.encoder.detector.class_embed.bias, 0)
            nn.init.constant_(self.decoder.detector.class_embed.bias, 0)

        nn.init.constant_(self.encoder.detector.bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(self.encoder.detector.bbox_embed.layers[-1].bias, 0)

        nn.init.constant_(self.decoder.detector.bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(self.decoder.detector.bbox_embed.layers[-1].bias, 0)

    @torch.jit.ignore
    def shard_modules(self):
        return {"encoder.layers", "decoder.layers"}

    def _create_ref_windows(self, tensor_list, mask_list):
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
            center = torch.stack([x_embed, y_embed], dim=-1).flatten(1, 2)

            h_embed = self.ref_size / size_h
            w_embed = self.ref_size / size_w

            size = torch.stack([w_embed, h_embed], dim=-1)
            size = size.unsqueeze(1).expand_as(center)

            ref_box = torch.cat([center, size], dim=-1)
            ref_windows.append(ref_box)

        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def _create_valid_ratios(self, src, masks):
        if masks is None:
            return None

        ratios = []
        for mask in masks:
            not_mask = ~mask
            size_h = not_mask[:, :, 0].sum(dim=-1, dtype=src[0].dtype)
            size_w = not_mask[:, 0, :].sum(dim=-1, dtype=src[0].dtype)

            h, w = mask.shape[-2:]
            ratio_w = size_w / w
            ratio_h = size_h / h
            ratio = torch.stack([ratio_w, ratio_h], dim=-1)

            ratios.append(ratio)
        valid_ratios = (
            torch.stack(ratios, dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(-2)
        )

        return valid_ratios

    def forward(self, src, mask, pos):
        if mask[0] is None:
            mask = None

        src_ref_windows = self._create_ref_windows(src, mask)
        src_valid_ratios = self._create_valid_ratios(src, mask)
        src, src_mask, src_shape = flatten_with_shape(src, mask)

        src_pos = []
        for pe in pos:
            b, c = pe.shape[:2]
            pe = pe.view(b, c, -1).transpose(1, 2)
            src_pos.append(pe)
        src_pos = torch.cat(src_pos, dim=1)

        src_start_index = torch.cat(
            [src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]]
        )

        output = self.encoder(
            src,
            src_pos,
            src_shape,
            src_mask,
            src_start_index,
            src_valid_ratios,
            src_ref_windows,
        )
        out_embed, dec_embed, dec_ref_windows, dec_pos, enc_out = output

        dec_out = self.decoder(
            dec_embed,
            dec_pos,
            out_embed,
            src_shape,
            src_mask,
            src_start_index,
            src_valid_ratios,
            dec_ref_windows,
        )

        if not self.inferencing:
            out = dec_out[-1]
            out["enc_outputs"] = [enc_out]
            out["aux_outputs"] = dec_out[:-1]

            return out

        return dec_out


class DeformableTransformerEncoder(nn.Module):
    def __init__(
        self,
        enc_dim,
        dec_dim,
        encoder_layer,
        encoder_detector,
        num_layers,
        num_queries,
        loss_mode,
        prenorm=False,
    ):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)

        if prenorm:
            self.norm = nn.LayerNorm(enc_dim)
        else:
            self.norm = nn.Identity()

        self.enc_linear = nn.Sequential(
            nn.Linear(enc_dim, dec_dim), nn.LayerNorm(dec_dim)
        )

        self.detector = encoder_detector
        self.num_queries = num_queries
        self.d_model = enc_dim
        self.loss_mode = loss_mode

    def _get_enc_proposals(self, output, src_mask, ref_windows):
        output_embed = output.detach()

        if self.loss_mode == "ce":
            out_logits = self.detector.class_embed(output_embed).softmax(dim=-1)[..., 0]
        else:
            out_logits = self.detector.class_embed(output_embed)[..., 0]
        if src_mask is not None:
            out_logits = out_logits.masked_fill(src_mask, -65504.0)

        num_queries = min(self.num_queries, out_logits.shape[1])
        _, indexes = torch.topk(out_logits, num_queries, dim=1, sorted=False)

        indexes = indexes.unsqueeze(-1)
        output_embed = torch.gather(
            output_embed, 1, indexes.expand(-1, -1, output.shape[-1])
        )

        ref_windows = torch.gather(ref_windows, 1, indexes.expand(-1, -1, 4))
        tmp_ref_windows = self.detector.bbox_embed(output_embed)
        tmp_ref_windows += inverse_sigmoid(ref_windows)
        out_ref_windows = tmp_ref_windows.sigmoid().detach()

        out_embed = self.enc_linear(output_embed)

        pos = get_proposal_pos_embed(out_ref_windows[..., :2], self.d_model)
        size = get_proposal_pos_embed(out_ref_windows[..., 2:], self.d_model)
        out_pos = pos + size

        return out_embed, out_ref_windows, out_pos

    def _get_enc_outputs(self, output, src_mask, ref_windows):
        return self.detector(output, ref_windows=ref_windows, x_mask=src_mask)

    def forward(
        self,
        src,
        pos,
        src_shape,
        src_mask,
        src_start_index,
        src_valid_ratios,
        ref_windows,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                pos,
                src_shape,
                src_mask,
                src_start_index,
                src_valid_ratios,
                ref_windows,
            )
        output = self.norm(output)

        if src_mask is not None:
            output = output.masked_fill(src_mask.unsqueeze(-1), 0.0)

        out_embed, out_ref_windows, out_pos = self._get_enc_proposals(
            output, src_mask, ref_windows
        )

        enc_out = self._get_enc_outputs(output, src_mask, ref_windows)

        return output, out_embed, out_ref_windows, out_pos, enc_out


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        decoder_layer,
        decoder_detector,
        num_layers,
        prenorm=False,
    ):
        super().__init__()

        self.layers = get_clones(decoder_layer, num_layers)
        if prenorm:
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.Identity()
        self.detector = decoder_detector

    def forward(
        self,
        tgt,
        query_pos,
        memory,
        memory_shape,
        memory_mask,
        memory_start_index,
        memory_valid_ratios,
        ref_windows,
    ):
        output = tgt
        dec_out = []

        for layer in self.layers:
            output = layer(
                output,
                query_pos,
                memory,
                memory_shape,
                memory_mask,
                memory_start_index,
                memory_valid_ratios,
                ref_windows,
            )

            if not self.inferencing:
                dec_out.append(
                    self.detector(
                        self.norm(output),
                        ref_windows=ref_windows,
                    )
                )

        if self.inferencing:
            dec_out = self.detector(
                self.norm(output),
                ref_windows=ref_windows,
            )

        return dec_out


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        nhead,
        nlevel,
        dim_feedforward,
        dropout,
        activation,
        prenorm=False,
    ):
        super().__init__()

        self.self_attn = DeformableAttention(dim, dim, nlevel, nhead)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.prenorm = prenorm

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        pos,
        src_shape,
        src_mask,
        src_start_index,
        src_valid_ratios,
        ref_points,
    ):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            src,
            src_shape,
            src_mask,
            src_start_index,
            src_valid_ratios,
            ref_points,
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(
        self,
        src,
        pos,
        src_shape,
        src_mask,
        src_start_index,
        src_valid_ratios,
        ref_points,
    ):
        src2 = self.norm1(src)
        src2 = self.self_attn(
            self.with_pos_embed(src2, pos),
            src2,
            src_shape,
            src_mask,
            src_start_index,
            src_valid_ratios,
            ref_points,
        )[0]
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src

    def forward(
        self,
        src,
        pos,
        src_shape,
        src_mask,
        src_start_index,
        src_valid_ratios,
        ref_points,
    ):
        if self.prenorm:
            return self.forward_pre(
                src,
                pos,
                src_shape,
                src_mask,
                src_start_index,
                src_valid_ratios,
                ref_points,
            )

        return self.forward_post(
            src,
            pos,
            src_shape,
            src_mask,
            src_start_index,
            src_valid_ratios,
            ref_points,
        )


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dec_dim,
        enc_dim,
        nhead,
        nlevel,
        dim_feedforward,
        dropout,
        activation,
        prenorm=False,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            dec_dim, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = DeformableAttention(dec_dim, enc_dim, nlevel, nhead)

        self.linear1 = nn.Linear(dec_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dec_dim)
        self.norm1 = nn.LayerNorm(dec_dim)
        self.norm2 = nn.LayerNorm(dec_dim)
        self.norm3 = nn.LayerNorm(dec_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.prenorm = prenorm

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        query_pos,
        memory,
        memory_shape,
        memory_mask,
        memory_start_index,
        memory_valid_ratios,
        ref_points,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            self.with_pos_embed(tgt, query_pos),
            memory,
            memory_shape,
            memory_mask,
            memory_start_index,
            memory_valid_ratios,
            ref_points,
        )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        query_pos,
        memory,
        memory_shape,
        memory_mask,
        memory_start_index,
        memory_valid_ratios,
        ref_points,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            self.with_pos_embed(tgt2, query_pos),
            memory,
            memory_shape,
            memory_mask,
            memory_start_index,
            memory_valid_ratios,
            ref_points,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        memory,
        memory_shape,
        memory_mask,
        memory_start_index,
        memory_valid_ratios,
        ref_points,
    ):
        if self.prenorm:
            return self.forward_pre(
                tgt,
                query_pos,
                memory,
                memory_shape,
                memory_mask,
                memory_start_index,
                memory_valid_ratios,
                ref_points,
            )

        return self.forward_post(
            tgt,
            query_pos,
            memory,
            memory_shape,
            memory_mask,
            memory_start_index,
            memory_valid_ratios,
            ref_points,
        )
