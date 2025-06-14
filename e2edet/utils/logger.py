# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmf (https://github.com/facebookresearch/mmf)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import collections
import json
import logging
import os
import sys
from functools import lru_cache
from typing import TypeVar

import torch
from torch.utils.tensorboard import SummaryWriter

from e2edet.utils.distributed import is_master
from e2edet.utils.timer import Timer


T = TypeVar("T", bound=type)

logger = logging.getLogger(__name__)


@lru_cache(None)
def log_once(logger: logging.Logger, msg: str, level: int = logging.INFO) -> None:
    """
    Logs a message only once. LRU cache is used to ensure a specific message is
    logged only once, similar to how :func:`~warnings.warn` works when the ``once``
    rule is set via command-line or environment variable.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    log_rank_zero(logger=logger, msg=msg, level=level)


def log_rank_zero(logger: logging.Logger, msg: str, level: int = logging.INFO) -> None:
    """
    Logs a message only on rank zero.

    Args:
        logger (logging.Logger): The logger.
        msg (str): The warning message.
        level (int): The logging level. See https://docs.python.org/3/library/logging.html#levels for values.
            Defaults to ``logging.INFO``.
    """
    if not is_master():
        return
    logger.log(level, msg)


class Logger:
    def __init__(self, save_dir, logger_level, log_format, should_not_log):
        self.logger = None
        self._is_master = is_master()

        self.timer = Timer()
        self.save_dir = save_dir
        self.debug = logger_level == "debug"

        self.log_format = log_format
        self.time_format = "%Y-%m-%dT%H:%M:%S"
        self.log_filename = "train_"
        self.log_filename += self.timer.get_time_hhmmss(None, format=self.time_format)
        self.log_filename += ".log"

        self.log_folder = os.path.join(self.save_dir, "logs")

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder, exist_ok=True)

        self.log_filename = os.path.join(self.log_folder, self.log_filename)

        if not self._is_master:
            return
        if self._is_master:
            print("Logging to:", self.log_filename)

        logging.captureWarnings(True)

        global logger
        self.logger = logger
        self._file_only_logger = logging.getLogger(__name__)
        warnings_logger = logging.getLogger("py.warnings")

        # Set level
        self.logger.setLevel(getattr(logging, logger_level.upper()))
        self._file_only_logger.setLevel(getattr(logging, logger_level.upper()))

        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )

        # Add handler to file
        channel = logging.FileHandler(filename=self.log_filename, mode="a")
        channel.setFormatter(formatter)

        self.logger.addHandler(channel)
        self._file_only_logger.addHandler(channel)
        warnings_logger.addHandler(channel)

        # Add handler to stdout
        channel = logging.StreamHandler(sys.stdout)
        channel.setFormatter(formatter)

        self.logger.addHandler(channel)
        warnings_logger.addHandler(channel)

        self.should_log = not should_not_log

        # Single log wrapper map
        self._single_log_map = set()

    def write(self, x, level="info", donot_print=False, log_all=False):
        if self.logger is None:
            return

        if log_all is False and not self._is_master:
            return

        # if it should not log then just print it
        if self.should_log:
            if hasattr(self.logger, level):
                if donot_print:
                    getattr(self._file_only_logger, level)(str(x))
                else:
                    getattr(self.logger, level)(str(x))
            else:
                self.logger.error("Unknown log level type: %s" % level)
        else:
            print(str(x) + "\n")

    def log_progress(self, info):
        if not isinstance(info, collections.abc.Mapping):
            self.write(info)

        if not self._is_master:
            return

        if self.log_format == "simple":
            output = ", ".join(
                ["{}: {}".format(key, value) for key, value in info.items()]
            )
        elif self.log_format == "json":
            output = json.dumps(info)
        else:
            output = str(info)

        self.write(output)

    def single_write(self, x, level="info", log_all=False):
        if self.logger is None:
            return
        if log_all is False and not self._is_master:
            return
        if x + "_" + level in self._single_log_map:
            return
        else:
            self.write(x, level)


class TensorboardLogger:
    def __init__(self, log_folder="./logs"):
        self.summary_writer = None
        self._is_master = is_master()
        self.log_folder = log_folder

        if self._is_master:
            self.summary_writer = SummaryWriter(self.log_folder)

    def __del__(self):
        if getattr(self, "summary_writer", None) is not None:
            self.summary_writer.close()

    def close(self):
        if getattr(self, "summary_writer", None) is not None:
            self.summary_writer.close()

    def _should_log_tensorboard(self):
        if self.summary_writer is None or not self._is_master:
            return False
        else:
            return True

    def add_scalar(self, key, value, iteration):
        if not self._should_log_tensorboard():
            return

        if isinstance(value, torch.Tensor):
            value = value.item()
        self.summary_writer.add_scalar(key, value, iteration)

    def add_scalars(self, scalar_dict, iteration):
        if not self._should_log_tensorboard():
            return

        for key, val in scalar_dict.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.summary_writer.add_scalar(key, val, iteration)

    def add_histogram_for_model(self, model, iteration):
        if not self._should_log_tensorboard():
            return

        for name, param in model.named_parameters():
            np_param = param.clone().cpu().data.numpy()
            self.summary_writer.add_histogram(name, np_param, iteration)

    def add_image(self, tag, image, iteration):
        if not self._should_log_tensorboard():
            return

        self.summary_writer.add_image(tag, image, iteration)

    def add_images(self, image_dict, iteration):
        if not self._should_log_tensorboard():
            return

        for tag, image in image_dict.items():
            self.summary_writer.add_image(tag, image, iteration)
