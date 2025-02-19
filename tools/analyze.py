import os
import argparse
from collections import Counter

import torch

from e2edet.utils.configuration import load_yaml
from e2edet.model import build_model
from e2edet.utils.general import get_root
from e2edet.utils.timer import Timer
from e2edet.dataset.helper import Prefetcher
from e2edet.dataset import build_dataset, build_dataloader

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])


def setup(args):
    config = load_yaml(args.config_path)
    config.training.batch_size = 1
    config.training.iter_per_update = 1
    config.training.num_workers = 32

    return config


def _multi_gpu_state_to_single(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if not k.startswith("module."):
            raise TypeError("Not a multiple GPU state of dict")
        k1 = k[7:]
        new_sd[k1] = v

    return new_sd


@torch.no_grad()
def do_speed(config):
    current_device = torch.device("cuda")
    dataset = build_dataset(config, "test", current_device)
    dataloader = build_dataloader(config, "test", dataset)[0]

    num_classes = dataset.get_answer_size()
    model = build_model(config, num_classes)

    if args.model_path is not None:
        ext = args.model_path.split(".")[-1]
        state_dict = torch.load(args.model_path, map_location="cpu")
        if ext == "ckpt":
            state_dict = state_dict["model"]

        if list(state_dict.keys())[0].startswith("module") and not hasattr(
            model, "module"
        ):
            state_dict = _multi_gpu_state_to_single(state_dict)

        print("Loading model:", model.load_state_dict(state_dict))
    model = model.to(current_device)
    # if TORCH_VERSION >= (2, 0):
    #     model = torch.compile(model, fullgraph=False)
    model.inference()

    timer = Timer()
    prefetcher = Prefetcher(dataloader, dataset, prefetch=True)

    if args.num_inputs > 0:
        start_idx = 50
    else:
        args.num_inputs = len(dataset)
        start_idx = 0

    for idx in range(args.num_inputs + start_idx):
        data = prefetcher.get_next_sample()
        if idx == start_idx:
            timer.reset()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(data[0])
            results = dataset.format_for_evalai(outputs, data[1])

    current_time = timer.unix_time_since_start()

    print(f"Time since start (seconds): {current_time}")
    print(f"Frame-per-second (fps): {args.num_inputs / current_time}")


def do_structure(config):
    current_device = torch.device("cuda")
    dataset = build_dataset(config, "test", current_device)

    num_classes = dataset.get_answer_size()
    model = build_model(config, num_classes)
    print("Model Structure:\n" + str(model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="""
Examples:
To show parameters of a model:
$ ./analyze_model.py --tasks parameter \\
    --config-path save/COCO-InstanceSegmentation/boxer2d_R_101_3x.yaml
Flops and activations are data-dependent, therefore inputs and model weights
are needed to count them:
$ ./analyze_model.py --num-inputs 100 --tasks flop \\
    --config-path save/COCO-InstanceSegmentation/boxer2d_R_101_3x.yaml \\
    --model-path save/COCO-InstanceSegmentation/boxer2d_final.pth
"""
    )
    parser.add_argument("--config-path", default=None, help="path to config file")
    parser.add_argument("--model-path", default=None, help="path to model file")
    parser.add_argument(
        "--tasks",
        choices=["flop", "parameter", "structure", "speed"],
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-n",
        "--num-inputs",
        default=-1,
        type=int,
        help="number of inputs used to compute statistics for flops/activations, "
        "both are data dependent.",
    )
    args = parser.parse_args()

    if args.config_path is not None and not os.path.isabs(args.config_path):
        args.config_path = os.path.join(get_root(), "..", args.config_path)

    if args.model_path is not None and not os.path.isabs(args.model_path):
        args.model_path = os.path.join(get_root(), "..", args.model_path)

    config = setup(args)
    for task in args.tasks:
        {
            "structure": do_structure,
            "speed": do_speed,
        }[task](config)
