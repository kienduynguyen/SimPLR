import warnings
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from ..attention.box_attention import (
    SimpleBoxAttention,
    SimpleInstanceAttention,
)
from e2edet.utils.general import (
    flatten_with_shape,
    inverse_sigmoid,
    get_clones,
    get_activation_fn,
    get_proposal_pos_embed,
)
from e2edet.module.head import MultiDetector, Detector
from e2edet.module.transformer import register_transformer


@register_transformer("simplr_transformer")
class SimPLRTransformer(nn.Module):
    def __init__(
        self,
        enc_dim=768,
        dec_dim=768,
        nhead=8,
        nlevel=1,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward_ratio=4.0,
        dropout=0.1,
        activation="relu",
        num_queries=300,
        num_classes=91,
        use_mask=False,
        ref_size=4,
        ref_size_ratios=[1],
        instance_mask=14,
        residual_mode="v1",
        loss_mode="focal",
        prenorm=False,
        dn_num=0,
        noise_scale=1,
        **kwargs,
    ):
        assert nlevel == 1
        super().__init__()

        if len(list(kwargs.keys())) > 0:
            warnings.warn(
                f"Arguments {list(kwargs.keys())} are unused in {self.__class__.__name__}"
            )

        if dn_num > 0:
            self.label_enc = nn.Embedding(num_classes, dec_dim)

        encoder_layer = SimPLREncoderLayer(
            enc_dim,
            nhead,
            enc_dim * dim_feedforward_ratio,
            dropout,
            activation,
            len(ref_size_ratios),
            prenorm=prenorm,
        )
        encoder_detector = MultiDetector(
            enc_dim, 1, len(ref_size_ratios), loss_mode=loss_mode
        )

        self.encoder = SimPLREncoder(
            enc_dim,
            dec_dim,
            encoder_layer,
            encoder_detector,
            enc_layers,
            num_queries,
            loss_mode,
            prenorm,
        )

        decoder_layer = SimPLRDecoderLayer(
            enc_dim,
            dec_dim,
            nhead,
            dec_dim * dim_feedforward_ratio,
            dropout,
            activation,
            use_mask,
            residual_mode,
            instance_mask,
            len(ref_size_ratios),
            prenorm=prenorm,
        )
        decoder_detector = Detector(
            dec_dim, num_classes, mask_mode="mask_v1", loss_mode=loss_mode
        )

        self.decoder = SimPLRDecoder(
            dec_dim,
            decoder_layer,
            decoder_detector,
            dec_layers,
            use_mask,
            prenorm=prenorm,
        )

        self.ref_size = ref_size
        self.ref_size_ratios = ref_size_ratios
        self.loss_mode = loss_mode
        self.dn_num = dn_num
        self.noise_scale = noise_scale
        self.num_classes = num_classes
        self.dec_dim = dec_dim
        self.num_queries = num_queries
        assert nhead % len(ref_size_ratios) == 0

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(
                m,
                (
                    SimpleBoxAttention,
                    SimpleInstanceAttention,
                ),
            ):
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
        nn.init.constant_(self.decoder.detector.mask_embed.layers[-1].bias, 0)

    @torch.jit.ignore
    def shard_modules(self):
        return {"encoder.layers", "decoder.layers"}

    def _create_ref_windows(self, tensor_list, mask_list):
        num_scale = len(self.ref_size_ratios)
        ref_size = self.ref_size * torch.FloatTensor(list(self.ref_size_ratios)).to(
            tensor_list[0]
        )
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
            center = center.unsqueeze(-2).expand(
                -1, -1, num_scale, -1
            )  # b x l x nh x 2

            h_embed = ref_size / size_h.unsqueeze(1)  # b x nh
            w_embed = ref_size / size_w.unsqueeze(1)  # b x nh

            size = torch.stack([w_embed, h_embed], dim=-1)  # b x nh x 2
            size = size.unsqueeze(1).expand_as(center)  # b x l x nh x 2

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

    def _post_process_for_dn(self, output, mask_dict):
        if mask_dict is not None:
            pad_size = mask_dict["pad_size"]
            assert pad_size > 0
        else:
            pad_size = 0

        pred_logits = output["pred_logits"]
        pred_boxes = output["pred_boxes"]

        outputs_class = pred_logits[:, pad_size:, :]
        outputs_dn_class = pred_logits[:, :pad_size, :]
        outputs_coord = pred_boxes[:, pad_size:, :]
        outputs_dn_coord = pred_boxes[:, :pad_size, :]

        out = {"pred_logits": outputs_class, "pred_boxes": outputs_coord}
        out_dn = {"pred_logits": outputs_dn_class, "pred_boxes": outputs_dn_coord}

        if "pred_masks" in output:
            pred_mask = output["pred_masks"]
            outputs_mask = pred_mask[:, pad_size:, :]
            outputs_dn_mask = pred_mask[:, :pad_size, :]

            out["pred_masks"] = outputs_mask
            out_dn["pred_masks"] = outputs_dn_mask

        return out, out_dn

    def _prepare_for_dn(self, targets, batch_size=1):
        if self.training:
            dn_num, noise_scale = self.dn_num, self.noise_scale

            known = [(torch.ones_like(t["labels"])) for t in targets]
            known_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num) > 0:
                dn_num = dn_num // (int(max(known_num)))
            else:
                dn_num = 0

            if dn_num == 0:
                query_label = None
                query_ref_windows = None
                query_pos = None
                attn_mask = None
                mask_dict = None

                return query_label, query_ref_windows, query_pos, attn_mask, mask_dict

            # can be modified to selectively denoise some labels or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t["labels"] for t in targets])
            boxes = torch.cat([t["boxes"] for t in targets])
            batch_idx = torch.cat(
                [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
            )
            # known
            known_indices = torch.nonzero(unmask_label + unmask_bbox)
            known_indices = known_indices.view(-1)

            # noise
            known_indices = known_indices.repeat(dn_num, 1).view(-1)
            known_labels = labels.repeat(dn_num, 1).view(-1)
            known_bid = batch_idx.repeat(dn_num, 1).view(-1)
            known_boxes = boxes.repeat(dn_num, 1)
            known_labels_expand = known_labels.clone()
            known_boxes_expand = known_boxes.clone()

            # noise
            if noise_scale > 0:
                label_p = torch.rand_like(known_labels_expand.float())
                chosen_indices = torch.nonzero(label_p < (noise_scale * 0.5)).view(
                    -1
                )  # half of bbox prob
                new_label = torch.randint_like(
                    chosen_indices, 0, self.num_classes
                )  # randomly put a new one here
                known_labels_expand.scatter_(0, chosen_indices, new_label)

                diff = torch.zeros_like(known_boxes_expand)
                diff[:, :2] = known_boxes_expand[:, 2:] / 2
                diff[:, 2:] = known_boxes_expand[:, 2:]
                boxes_offset = torch.rand_like(known_boxes_expand) * 2 - 1.0

                known_boxes_expand += torch.mul(boxes_offset, diff) * noise_scale
                known_boxes_expand = known_boxes_expand.clamp(min=0.0, max=1.0)

            labels_index = known_labels_expand.long()
            labels_embed = self.label_enc(labels_index)
            boxes_embed = known_boxes_expand

            single_pad = int(max(known_num))
            pad_size = int(single_pad * dn_num)

            query_label = torch.zeros(
                pad_size,
                self.dec_dim,
                dtype=self.label_enc.weight.dtype,
                device=self.label_enc.weight.device,
            ).repeat(batch_size, 1, 1)
            query_ref_windows = torch.zeros(
                pad_size,
                4,
                dtype=self.label_enc.weight.dtype,
                device=self.label_enc.weight.device,
            ).repeat(batch_size, 1, 1)

            # if dec_embed is not None:
            #     query_label = torch.cat([padding_label, dec_embed], dim=0).repeat(
            #         batch_size, 1, 1
            #     )
            #     query_ref_windows = torch.cat(
            #         [padding_box, dec_ref_windows], dim=0
            #     ).repeat(batch_size, 1, 1)
            # else:
            #     query_label = padding_label
            #     query_ref_windows = padding_box

            # map
            map_known_indices = torch.tensor([]).to(self.label_enc.weight)
            if len(known_num):
                map_known_indices = torch.cat(
                    [torch.tensor(range(num)) for num in known_num]
                )
                map_known_indices = torch.cat(
                    [map_known_indices + single_pad * i for i in range(dn_num)]
                ).long()
            if len(known_bid):
                query_label[(known_bid.long(), map_known_indices)] = labels_embed
                query_ref_windows[(known_bid.long(), map_known_indices)] = boxes_embed
            pos = get_proposal_pos_embed(query_ref_windows[..., :2], self.dec_dim)
            size = get_proposal_pos_embed(query_ref_windows[..., 2:], self.dec_dim)
            query_pos = pos + size

            query_size = pad_size + self.num_queries
            attn_mask = torch.zeros(
                query_size,
                query_size,
                dtype=torch.bool,
                device=self.label_enc.weight.device,
            )
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(dn_num):
                if i == 0:
                    attn_mask[
                        single_pad * i : single_pad * (i + 1),
                        single_pad * (i + 1) : pad_size,
                    ] = True
                elif i == dn_num - 1:
                    attn_mask[
                        single_pad * i : single_pad * (i + 1), : single_pad * i
                    ] = True
                else:
                    attn_mask[
                        single_pad * i : single_pad * (i + 1),
                        single_pad * (i + 1) : pad_size,
                    ] = True
                    attn_mask[
                        single_pad * i : single_pad * (i + 1), : single_pad * i
                    ] = True
            mask_dict = {
                "known_indices": torch.as_tensor(known_indices).long(),
                "batch_idx": torch.as_tensor(batch_idx).long(),
                "map_known_indices": torch.as_tensor(map_known_indices).long(),
                "known_lbs_bboxes": (known_labels, known_boxes),
                "know_idx": known_idx,
                "pad_size": pad_size,
                "scalar": dn_num,
            }
        else:
            query_label = None
            query_ref_windows = None
            query_pos = None
            attn_mask = None
            mask_dict = None

        return query_label, query_ref_windows, query_pos, attn_mask, mask_dict

    def forward(self, src, mask, pos, targets=None):
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

        (
            dn_embed,
            dn_ref_windows,
            dn_pos,
            dn_attn_mask,
            mask_dict,
        ) = self._prepare_for_dn(targets, batch_size=out_embed.shape[0])

        if mask_dict is not None:
            dec_embed = torch.cat([dn_embed, dec_embed], dim=1)
            dec_ref_windows = torch.cat([dn_ref_windows, dec_ref_windows], dim=1)
            dec_pos = torch.cat([dn_pos, dec_pos], dim=1)

        dec_out = self.decoder(
            dec_embed,
            dec_pos,
            out_embed,
            src_shape,
            src_mask,
            src_start_index,
            src_valid_ratios,
            dec_ref_windows,
            dn_attn_mask,
        )

        if not self.inferencing:
            if self.dn_num > 0:
                dn_outputs = []
                aux_outputs = []

                output, out_dn = self._post_process_for_dn(dec_out[-1], mask_dict)
                dn_outputs.append(out_dn)
                for elem in dec_out[:-1]:
                    out, out_dn = self._post_process_for_dn(elem, mask_dict)
                    aux_outputs.append(out)
                    dn_outputs.append(out_dn)
                output["aux_outputs"] = aux_outputs
                output["dn_outputs"] = dn_outputs
                output["mask_dict"] = mask_dict

                # hack for guaranteed gradient
                output["pred_logits"] += 0.0 * self.label_enc.weight.sum()
            else:
                output = dec_out[-1]
                output["aux_outputs"] = dec_out[:-1]
            output["enc_outputs"] = [enc_out]

            return output

        return dec_out


class SimPLREncoder(nn.Module):
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
        self.loss_mode = loss_mode
        self.d_model = dec_dim

    def _get_enc_proposals(self, output, src_mask, ref_windows):
        b, l = output.shape[:2]
        output_embed = output.detach()
        num_scale = self.detector.num_references

        tmp_ref_windows = self.detector.bbox_embed(output_embed)
        tmp_ref_windows = tmp_ref_windows.view(b, l, num_scale, -1)

        out_ref_windows = tmp_ref_windows + inverse_sigmoid(ref_windows)
        out_ref_windows = out_ref_windows.view(b, l * num_scale, -1)

        if self.loss_mode == "ce":
            out_logits = (
                self.detector.class_embed(output_embed)
                .view(b, l, num_scale, -1)
                .softmax(dim=-1)[..., 0]
            )
        else:
            out_logits = self.detector.class_embed(output_embed).view(
                b, l, num_scale, -1
            )[..., 0]

        if src_mask is not None:
            out_logits = out_logits.masked_fill(src_mask.unsqueeze(-1), -65504.0)
        out_logits = out_logits.view(b, l * num_scale)

        num_queries = min(self.num_queries, out_logits.shape[1])
        _, indexes = torch.topk(out_logits, num_queries, dim=1, sorted=False)

        indexes = indexes.unsqueeze(-1)
        out_ref_windows = torch.gather(
            out_ref_windows,
            1,
            index=indexes.expand(-1, -1, out_ref_windows.shape[-1]),
        )
        out_ref_windows = out_ref_windows.sigmoid().detach()

        pos = get_proposal_pos_embed(out_ref_windows[..., :2], self.d_model)
        size = get_proposal_pos_embed(out_ref_windows[..., 2:], self.d_model)
        out_pos = pos + size

        indexes = indexes.expand(-1, -1, output.shape[-1]).div(
            num_scale, rounding_mode="floor"
        )
        output_embed = torch.gather(output_embed, 1, indexes)
        out_embed = self.enc_linear(output_embed)

        return out_embed, out_ref_windows, out_pos

    def _get_enc_outputs(self, output, src_mask, ref_windows):
        return self.detector(output[None], ref_windows=ref_windows, x_mask=src_mask)

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

        for i, layer in enumerate(self.layers):
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

        enc_out = None
        if not self.inferencing:
            enc_out = self._get_enc_outputs(output, src_mask, ref_windows)

        return output, out_embed, out_ref_windows, out_pos, enc_out


class SimPLRDecoder(nn.Module):
    def __init__(
        self,
        dec_dim,
        decoder_layer,
        decoder_detector,
        num_layers,
        use_mask,
        prenorm=False,
    ):
        super().__init__()

        self.layers = get_clones(decoder_layer, num_layers)
        if prenorm:
            self.norm = nn.LayerNorm(dec_dim)
        else:
            self.norm = nn.Identity()
        self.detector = decoder_detector
        self.use_mask = use_mask

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
        dn_attn_mask=None,
    ):
        output = tgt
        dec_out = []

        for i, layer in enumerate(self.layers):
            # hack to return mask from the last layer
            if i == len(self.layers) - 1:
                layer.inferencing = False
                layer.multihead_attn.inferencing = False

            output, roi_feat = layer(
                output,
                query_pos,
                memory,
                memory_shape,
                memory_mask,
                memory_start_index,
                memory_valid_ratios,
                ref_windows,
                dn_attn_mask,
            )

            if not self.inferencing:
                dec_out.append(
                    self.detector(
                        self.norm(output),
                        ref_windows=ref_windows,
                        roi=self.norm(roi_feat) if roi_feat is not None else roi_feat,
                    )
                )

        if self.inferencing:
            dec_out = self.detector(
                self.norm(output),
                ref_windows=ref_windows,
                roi=self.norm(roi_feat) if roi_feat is not None else roi_feat,
            )

        return dec_out


class SimPLREncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        num_box,
        prenorm=False,
    ):
        super().__init__()

        self.self_attn = SimpleBoxAttention(d_model, d_model, num_box, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

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
        ref_windows,
    ):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            src,
            src_shape,
            src_mask,
            src_start_index,
            src_valid_ratios,
            ref_windows,
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
        ref_windows,
    ):
        src2 = self.norm1(src)
        src2 = self.self_attn(
            self.with_pos_embed(src2, pos),
            src2,
            src_shape,
            src_mask,
            src_start_index,
            src_valid_ratios,
            ref_windows,
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
        ref_windows,
    ):
        if self.prenorm:
            return self.forward_pre(
                src,
                pos,
                src_shape,
                src_mask,
                src_start_index,
                src_valid_ratios,
                ref_windows,
            )

        return self.forward_post(
            src,
            pos,
            src_shape,
            src_mask,
            src_start_index,
            src_valid_ratios,
            ref_windows,
        )


class SimPLRDecoderLayer(nn.Module):
    def __init__(
        self,
        enc_dim,
        dec_dim,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        use_mask,
        residual_mode,
        instance_mask,
        num_box,
        prenorm=False,
    ):
        super().__init__()
        assert residual_mode in ("v1", "v2")

        self.self_attn = nn.MultiheadAttention(
            dec_dim, 8, dropout=dropout, batch_first=True
        )
        if use_mask:
            self.multihead_attn = SimpleInstanceAttention(
                dec_dim, enc_dim, num_box, nhead, instance_mask
            )
        else:
            self.multihead_attn = SimpleBoxAttention(dec_dim, enc_dim, num_box, nhead)

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
        self.residual_mode = residual_mode
        self.use_mask = use_mask
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
        ref_windows,
        dn_attn_mask=None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=dn_attn_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.use_mask and not self.inferencing:
            tgt2, roi = self.multihead_attn(
                self.with_pos_embed(tgt, query_pos),
                memory,
                memory_shape,
                memory_mask,
                memory_start_index,
                memory_valid_ratios,
                ref_windows,
            )[:2]
        else:
            tgt2 = self.multihead_attn(
                self.with_pos_embed(tgt, query_pos),
                memory,
                memory_shape,
                memory_mask,
                memory_start_index,
                memory_valid_ratios,
                ref_windows,
            )[0]
            roi = None

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if self.use_mask and not self.inferencing:
            roi = tgt.unsqueeze(-2).unsqueeze(-2) + self.dropout2(roi)
            roi = self.norm2(roi)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if self.use_mask and not self.inferencing:
            if self.residual_mode == "v1":
                roi2 = self.linear2(self.dropout(self.activation(self.linear1(roi))))
                roi = roi + self.dropout3(roi2)
            elif self.residual_mode == "v2":
                roi = roi + self.dropout3(tgt.unsqueeze(-2).unsqueeze(-2))
            roi = self.norm3(roi)

        return tgt, roi

    def forward_pre(
        self,
        tgt,
        query_pos,
        memory,
        memory_shape,
        memory_mask,
        memory_start_index,
        memory_valid_ratios,
        ref_windows,
        dn_attn_mask=None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2, attn_mask=dn_attn_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        if self.use_mask and not self.inferencing:
            tgt2, roi = self.multihead_attn(
                self.with_pos_embed(tgt2, query_pos),
                memory,
                memory_shape,
                memory_mask,
                memory_start_index,
                memory_valid_ratios,
                ref_windows,
            )[:2]
        else:
            tgt2 = self.multihead_attn(
                self.with_pos_embed(tgt2, query_pos),
                memory,
                memory_shape,
                memory_mask,
                memory_start_index,
                memory_valid_ratios,
                ref_windows,
            )[0]
            roi = None
        tgt = tgt + self.dropout2(tgt2)
        if self.use_mask and not self.inferencing:
            roi = tgt.unsqueeze(-2).unsqueeze(-2) + self.dropout2(roi)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if self.use_mask and not self.inferencing:
            if self.residual_mode == "v1":
                roi2 = self.norm3(roi)
                roi2 = self.linear2(self.dropout(self.activation(self.linear1(roi2))))
                roi = roi + self.dropout3(roi2)
            elif self.residual_mode == "v2":
                roi = roi + self.dropout2(tgt2.unsqueeze(-2).unsqueeze(-2))

        return tgt, roi

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
        dn_attn_mask=None,
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
                ref_windows,
                dn_attn_mask=dn_attn_mask,
            )

        return self.forward_post(
            tgt,
            query_pos,
            memory,
            memory_shape,
            memory_mask,
            memory_start_index,
            memory_valid_ratios,
            ref_windows,
            dn_attn_mask=dn_attn_mask,
        )
