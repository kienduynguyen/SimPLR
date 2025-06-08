# ------------------------------------------------------------------------
# BoxeR
# Copyright (c) 2022. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmf (https://github.com/facebookresearch/mmf)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import numpy as np
import socket
import subprocess
import warnings
import datetime
from functools import lru_cache

import torch
import torch.distributed.nn
from torch import distributed as dist


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@lru_cache
def get_data_world_size():
    global_world_size = get_world_size()

    return int(os.environ.get("DATA_WORLD_SIZE", global_world_size))


def get_rank():
    if not dist.is_nccl_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size_and_rank():
    return get_world_size(), get_rank()


_LOCAL_RANK = None


def get_local_rank():
    global _LOCAL_RANK

    if _LOCAL_RANK is None:
        return 0

    return _LOCAL_RANK


def is_master():
    return get_rank() == 0


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not dist.is_nccl_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def broadcast_tensor(tensor, src=0):
    world_size = get_world_size()
    if world_size < 2:
        return tensor

    with torch.no_grad():
        dist.broadcast(tensor, src=0)

    return tensor


def broadcast_scalar(scalar, src=0, device="cpu"):
    scalar_tensor = torch.tensor(scalar).to(device)
    scalar_tensor = broadcast_tensor(scalar_tensor, src)
    return scalar_tensor.item()


def reduce_tensor(tensor):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if dist.get_rank() == 0:
            tensor = tensor.div(world_size)

    return tensor


def gather_tensor(tensor):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        tensor_list = []

        for _ in range(world_size):
            tensor_list.append(torch.zeros_like(tensor))

        dist.all_gather(tensor_list, tensor)
        tensor_list = torch.stack(tensor_list, dim=0)
    return tensor_list


def gather_features(features, gather_with_grad=False):
    world_size = get_world_size()

    if world_size < 2:
        return features

    if gather_with_grad:
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
    else:
        all_features = []

        for _ in range(world_size):
            all_features.append(torch.zeros_like(features))

        dist.all_gather(all_features, features)
        all_features = torch.cat(all_features, dim=0)

    return all_features


def reduce_dict(dictionary):
    world_size = get_world_size()
    if world_size < 2:
        return dictionary

    with torch.no_grad():
        keys, values = zip(*sorted(dictionary.items()))
        values = torch.stack(values, dim=0)

        dist.reduce(values, dst=0)

        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(keys, values)}
    return reduced_dict


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def all_reduce_dict(dictionary):
    world_size = get_world_size()
    if world_size < 2:
        return dictionary

    with torch.no_grad():
        keys, values = zip(*sorted(dictionary.items()))
        values = torch.stack(values, dim=0)

        dist.all_reduce(values)
        values /= world_size
        reduced_dict = {k: v for k, v in zip(keys, values)}

    return reduced_dict


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = dist.group.WORLD  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)

    return output


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def infer_init_method(config):
    if config.distributed.init_method is not None:
        return

    # support torch.distributed.launch
    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        print("support launch")
        config.distributed.init_method = "env://"
        config.distributed.world_size = int(os.environ["WORLD_SIZE"])
        config.distributed.rank = int(os.environ["RANK"])

    elif all(
        key in os.environ for key in ["ARNOLD_NUM", "ARNOLD_WORKER_0_HOST", "ARNOLD_ID"]
    ):
        config.distributed.port = int(os.environ.get("ARNOLD_WORKER_0_PORT"))

        nnodes = int(os.environ.get("ARNOLD_NUM"))
        if nnodes == 1 and torch.cuda.device_count() == 1:
            return

        config.distributed.init_method = "tcp://[{host}]:{port}".format(
            host=os.environ.get("ARNOLD_WORKER_0_HOST"),
            port=config.distributed.port,
        )

        gpus_per_node = torch.cuda.device_count()
        config.distributed.world_size = nnodes * gpus_per_node

        node_id = int(os.environ.get("ARNOLD_ID"))
        config.distributed.rank = node_id * gpus_per_node

    # we can determine the init method automatically for Slurm
    else:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            if config.distributed.port < 0:
                config.distributed.port = 16749
            try:
                nnodes = int(os.environ.get("SLURM_NNODES"))

                # don't need to initialize distributed training on a single gpu
                if nnodes == 1 and torch.cuda.device_count() == 1:
                    return

                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", node_list]
                )
                config.distributed.init_method = "tcp://{host}:{port}".format(
                    host=hostnames.split()[0].decode("utf-8"),
                    port=config.distributed.port,
                )

                ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get("SLURM_NTASKS"))
                    assert ntasks % nnodes == 0, f"ntasks: {ntasks}, nnodes: {nnodes}"
                    ntasks_per_node = int(ntasks / nnodes)

                gpus_per_node = torch.cuda.device_count()
                config.distributed.world_size = nnodes * gpus_per_node
                if ntasks_per_node == 1:
                    node_id = int(os.environ.get("SLURM_NODEID"))
                    config.distributed.rank = node_id * gpus_per_node
                else:
                    assert (
                        gpus_per_node == ntasks_per_node
                    ), f"gpus_per_node: {gpus_per_node}, ntasks_per_node: {ntasks_per_node}"
                    config.distributed.no_spawn = True
                    config.distributed.rank = int(os.environ.get("SLURM_PROCID"))
                    config.device_id = int(os.environ.get("SLURM_LOCALID"))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass


TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
SKIP_CLEANUP = "3"
BUFFER_SIZE = "20000"


def _warn_overwrite_env(env, val):
    if env in os.environ:
        print(
            f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config"
        )
    os.environ[env] = val


def distributed_init(config):
    # # FlightRecorder is incompatible with =1 mode where watchdog aborts work, must use =3 (skipcleanup)
    # # to get flight recorder dumps. See https://github.com/pytorch/pytorch/issues/121055
    # # This could be done only when flight recorder is enabled, but its nice to be consistent to avoid subtle
    # # behavior differences
    # _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)

    # # enable torch nccl flight recorder in the mode that would dump files if timeout is detected
    # _warn_overwrite_env(TRACE_BUFFER_SIZE, BUFFER_SIZE)
    # # dump on timeout by default if trace buffer is enabled
    # _warn_overwrite_env(DUMP_ON_TIMEOUT, "1")
    # dump_dir = f"{config.training.save_dir}/comm_trace"
    # os.makedirs(dump_dir, exist_ok=True)
    # _warn_overwrite_env(TRACE_FILE, f"{dump_dir}/rank_")

    # # to mitigate the memory issue that collectives using
    # # async_op=True hold memory longer than they should
    # # such as those in tensor parallelism
    # os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    if config.distributed.world_size == 1:
        raise ValueError("Cannot initialize distributed with distributed_world_size=1")

    timeout_duration = datetime.timedelta(minutes=60)

    if dist.is_initialized():
        warnings.warn("Distributed is already initialized, cannot initialize twice!")
    else:
        print(
            "Distributed Init (Rank {}): {}".format(
                config.distributed.rank, config.distributed.init_method
            ),
            flush=True,
        )

        global _LOCAL_RANK
        _LOCAL_RANK = config.device_id

        dist.init_process_group(
            backend=config.distributed.backend,
            init_method=config.distributed.init_method,
            world_size=config.distributed.world_size,
            rank=config.distributed.rank,
            timeout=timeout_duration,
        )
        print(
            "Initialized Host {} as Rank {}".format(
                socket.gethostname(), config.distributed.rank
            ),
            flush=True,
        )

        # perform a dummy all-reduce to initialize the NCCL communicator
        dist.all_reduce(torch.zeros(1).cuda())

        # suppress_output(is_master())

    config.distributed.rank = dist.get_rank()
    return config.distributed.rank


def destroy_process_group():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    import warnings

    builtin_warn = warnings.warn

    def warn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_warn(*args, **kwargs)

    # Log warnings only once
    warnings.warn = warn
    warnings.simplefilter("once", UserWarning)
