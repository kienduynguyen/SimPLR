import importlib
import os

from .base_engine import BaseEngine


TRAINER_REGISTRY = {}
ENGINE_REGISTRY = {}


def build_trainer(configuration, *rest, **kwargs):
    configuration.freeze()

    config = configuration.get_config()
    trainer = config.training.trainer
    trainer = TRAINER_REGISTRY[trainer](configuration)

    return trainer


def build_engine(config, trainer):
    task_name = config.task
    engine = ENGINE_REGISTRY[task_name](trainer)

    return engine


def register_trainer(name):
    def register_trainer_cls(cls):
        if name in TRAINER_REGISTRY:
            raise ValueError("Cannot register duplicate trainer ({})".format(name))

        TRAINER_REGISTRY[name] = cls
        return cls

    return register_trainer_cls


def register_engine(*name_list):
    def register_engine_cls(cls):
        for name in name_list:
            if name in ENGINE_REGISTRY:
                raise ValueError("Cannot register duplicate engine ({})".format(name))
            elif not issubclass(cls, BaseEngine):
                raise ValueError(
                    "Engine ({}: {}) must extend BaseEngine".format(name, cls.__name__)
                )

            ENGINE_REGISTRY[name] = cls
        return cls

    return register_engine_cls


trainers_dir = os.path.dirname(__file__)
for file in os.listdir(trainers_dir):
    path = os.path.join(trainers_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        trainer_name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("e2edet.trainer." + trainer_name)
