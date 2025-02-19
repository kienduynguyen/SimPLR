import importlib
import os

from .position_encoding import build_position_encoding


TRANSFORMER_REGISTRY = {}


__all__ = ["build_transformer", "build_position_encoding"]


def build_transformer(config):
    arch = config.type
    params = config.params

    if arch not in TRANSFORMER_REGISTRY:
        raise ValueError(f"Transformer architecture ({arch}) is not found")

    transformer = TRANSFORMER_REGISTRY[arch](**params)

    return transformer


def register_transformer(name):
    def register_transformer_cls(cls):
        if name in TRANSFORMER_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))

        TRANSFORMER_REGISTRY[name] = cls
        return cls

    return register_transformer_cls


transformers_dir = os.path.dirname(__file__)
for file in os.listdir(transformers_dir):
    path = os.path.join(transformers_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        transformer_name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("e2edet.module.transformer." + transformer_name)
