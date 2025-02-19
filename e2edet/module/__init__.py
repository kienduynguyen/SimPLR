from e2edet.module.matcher import build_matcher
from e2edet.module.head import (
    Detector,
    MultiDetector3d,
    Detector3d,
    Segmentor,
    MultiDetector,
)
from e2edet.module.backbone import build_resnet, build_backbone3d
from e2edet.module.transformer import build_transformer
from e2edet.module.vit import (
    build_vit_det,
    build_vit_det_fast,
    build_swin,
    LayerNorm,
)


def build_backbone2d(config):
    if "swin" in config["type"]:
        return build_swin(config)
    else:
        return build_resnet(config)


__all__ = [
    "build_matcher",
    "build_vit_det",
    "build_vit_det_fast",
    "build_resnet",
    "build_transformer",
    "build_backbone3d",
    "build_backbone2d",
    "Detector",
    "MultiDetector3d",
    "MultiDetector",
    "Detector3d",
    "Segmentor",
    "LayerNorm",
]
