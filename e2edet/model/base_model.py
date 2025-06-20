import torch
from torch import nn


class BaseDetectionModel(nn.Module):
    """For integration with Pythia's trainer, datasets and other features,
    models needs to inherit this class, call `super`, write a build function,
    write a forward function taking a ``SampleList`` as input and returning a
    dict as output and finally, register it using ``@registry.register_model``

    Args:
        config (DictConfig): ``model_config`` configuration from global config.

    """

    def __init__(self, config, global_config):
        super().__init__()
        self.config = config
        self._global_config = global_config

    def _build(self):
        """Function to be implemented by the child class, in case they need to
        build their model separately than ``__init__``. All model related
        downloads should also happen here.
        """
        raise NotImplementedError(
            "Build method not implemented in the child model class."
        )

    @torch.jit.ignore
    def shard_modules(self):
        shard_modules = set()

        for shard_module in self.backbone.shard_modules():
            shard_modules.add("backbone." + shard_module)

        for shard_module in self.transformer.shard_modules():
            shard_modules.add("transformer." + shard_module)

        return shard_modules

    def build(self):
        self._build()
        self.inference(False)

    def inference(self, mode=True):
        if mode:
            super().train(False)
        self.inferencing = mode
        for module in self.modules():
            if hasattr(module, "inferencing"):
                module.inferencing = mode
            else:
                setattr(module, "inferencing", mode)

    def train(self, mode=True):
        if mode:
            self.inferencing = False
            for module in self.modules():
                if hasattr(module, "inferencing"):
                    module.inferencing = False
                else:
                    setattr(module, "inferencing", False)
        super().train(mode)
