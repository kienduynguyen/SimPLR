import math
import copy

import omegaconf

from e2edet.model import BaseDetectionModel, register_model
from e2edet.module import build_vit_det, build_transformer
from e2edet.utils.modeling import get_parameters, get_vit_parameters
from e2edet.utils.distributed import synchronize


@register_model("deformable_detr_vit")
class DeformableDETRViT(BaseDetectionModel):
    def __init__(self, config, num_classes, global_config):
        super().__init__(config, global_config)
        self.deform_lr_multi = config["deform_lr_multi"]
        self.num_level = config["transformer"]["params"]["nlevel"]
        self.num_classes = num_classes

    def get_optimizer_parameters(self):
        lr_decay_rate = self._global_config.optimizer.params.lr_decay_rate

        backbone_groups = get_vit_parameters(
            self.backbone,
            lr_decay_rate=lr_decay_rate,
            wd_norm=0.0,
            wd_except=["pos_embed"],
            num_layers=self.backbone.net.depth,
        )

        transformer_groups = get_parameters(
            self,
            lr_multi=self.deform_lr_multi,
            lr_module=["sampling_offsets"],
            wd_norm=0.0,
            module_except=["backbone"],
        )

        return (backbone_groups, transformer_groups)

    def _build(self):
        self.backbone = build_vit_det(self.config.backbone)

        transformer_config = copy.deepcopy(self.config.transformer)
        with omegaconf.open_dict(transformer_config):
            transformer_config["params"]["num_classes"] = self.num_classes
        self.transformer = build_transformer(transformer_config)

    def forward(self, sample, target=None):
        outputs = self.backbone(sample["image"], sample["mask"])

        features = []
        masks = []
        pos_encodings = []

        for i, feature_name in enumerate(self.backbone._out_features):
            f, f_mask, f_pos = outputs[feature_name]
            features.append(f)
            pos_encodings.append(f_pos)
            masks.append(f_mask)

        out = self.transformer(features, masks, pos_encodings)

        return out
