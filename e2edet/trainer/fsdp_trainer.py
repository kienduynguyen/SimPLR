import os

import torch
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

torch._dynamo.config.cache_size_limit = 50

from e2edet.utils.meter import Meter
from e2edet.utils.distributed import is_master, get_world_size
from e2edet.utils.general import (
    print_model_parameters,
    get_batch_size,
    get_root,
    cleanup_before_training,
)
from e2edet.utils.logger import TensorboardLogger
from e2edet.utils.timer import Timer
from e2edet.trainer import register_trainer
from e2edet.trainer.engine import build_engine
from e2edet.model import build_model
from e2edet.optim import build_optimizer
from e2edet.optim.scheduler import build_scheduler
from e2edet.dataset import build_dataset, build_dataloader
from e2edet.module.parallelize import (
    ParallelDims,
    CheckpointManager,
    parallelize_model,
    reduce_dict,
    set_determinism,
)
from e2edet.module.quantization import Float8Converter
from e2edet.utils.logger import logger

from .base_trainer import BaseTrainer


@register_trainer("fsdp_trainer")
class FSDPTrainer(BaseTrainer):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.parallel_config = self.config.parallel
        self.parallel_dims = ParallelDims(
            dp_shard=self.parallel_config.data_parallel_shard_degree,
            dp_replicate=self.parallel_config.data_parallel_replicate_degree,
            tp=self.parallel_config.tensor_parallel_degree,
            world_size=get_world_size(),
            enable_loss_parallel=False,
        )
        self.world_mesh = self.parallel_dims.build_mesh(device_type="cuda")

        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0
        self.dp_degree = dp_degree
        self.dp_rank = dp_rank
        self.dp_mesh = dp_mesh

        self.use_float8 = self.config.quantization.use_float8

        os.environ["DATA_WORLD_SIZE"] = str(dp_degree)
        self._set_device()
        seed = self.running_config.seed
        set_determinism(self.world_mesh, self.device, seed)
        cleanup_before_training()

    @property
    def model_without_ddp(self):
        return self.model

    def _load_split_task(self, splits):
        self.splits = []
        for split in splits:
            dataset, dataloader, sampler = None, None, None
            if split in self.run_type:
                dataset = build_dataset(self.config, split, self.device)
                dataloader, sampler = build_dataloader(
                    self.config,
                    split,
                    dataset,
                    rank=self.dp_rank,
                    world_size=self.dp_degree,
                )
                self.splits.append(split)

            self.datasets[split] = dataset
            self.dataloaders[split] = dataloader
            self.samplers[split] = sampler

        for split, dataset in self.datasets.items():
            if dataset is not None:
                print(f"{split}: {len(dataset)} images")
                print(f"{split}: {dataset}")

    def _parallelize_model(self):
        self.parallel = True

        if self.use_float8:
            model_converter = Float8Converter(
                self.config.quantization.enable_fsdp_float8_all_gather,
                self.config.quantization.precompute_float8_dynamic_scale_for_fsdp,
                self.config.quantization.force_recompute_fp8_weight_in_bwd,
                self.parallel_dims,
                recipe_name=self.config.quantization.recipe_name,
                filter_fqns=self.config.quantization.filter_fqns,
            )
            model_converter.convert(self.model)

        parallelize_model(
            self.model,
            self.world_mesh,
            self.parallel_dims,
            self.parallel_config.use_compile,
            self.parallel_config.enable_async_tp,
            self.parallel_config.ac_mode,
            self.parallel_config.selective_ac_option,
            self.parallel_config.mp_param,
            self.parallel_config.mp_reduce,
            self.parallel_config.enable_compiled_autograd,
            self.parallel_config.cpu_offload,
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        init_device = "cpu" if self.parallel_config.create_seed_checkpoint else "cuda"
        self.model.to(device=init_device)

    def load_model_and_optimizer(self):
        self.writer.write("Loading model and optimizer", "info")

        if hasattr(self.datasets[self.splits[0]], "get_model_params"):
            model_params = self.datasets[self.splits[0]].get_model_params()
        else:
            model_params = {}

        # with torch.device("meta"):
        self.model = build_model(self.config, **model_params)

        if (
            self.running_config.resume
            and self.running_config.resume_file
            and not self.running_config.resume_dcp
        ):
            resume_file = self.running_config.resume_file
            if not os.path.isabs(resume_file):
                resume_file = os.path.normpath(
                    os.path.join(get_root(), "..", resume_file)
                )

            logger.info(f"Loading full_state_dict model only from {resume_file}")
            state_dict = torch.load(resume_file, map_location="cpu", weights_only=True)
            incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
            print("Model loaded:", incompatible_keys)

        print_model_parameters(self.model, self.writer)

        if "cuda" in str(self.device):
            device_info = "CUDA Device {} is: {}".format(
                self.config.distributed.rank,
                torch.cuda.get_device_name(self.local_rank),
            )
            self.writer.write(device_info, log_all=True)

        self.max_update = self.running_config.max_update
        self.max_epoch = self.running_config.max_epoch

        if self.max_epoch is not None and self.max_update is not None:
            raise ValueError("max_epoch and max_update are mutually exclusive!")

        if self.dataloaders["train"] is not None:
            update_per_epoch = len(self.dataloaders["train"])

            if self.max_epoch is not None:
                self.max_update = self.max_epoch * update_per_epoch
        else:
            self.max_update = 0

        self._parallelize_model()
        self.writer.write("===== Model =====")
        self.writer.write(self.model)
        
        self.optimizer = build_optimizer(self.config, self.model)
        if self.config.scheduler.type == "cosine_annealing":
            self.config.scheduler.params.T_max = self.max_update
        self.lr_scheduler = build_scheduler(self.config, self.optimizer)
        self._init_params_and_checkpoint()
        self._init_losses_and_metrics()

    def _init_params_and_checkpoint(self):
        self.writer.write("Torch version is: " + torch.__version__)

        self.current_epoch = 0
        self.current_update = 0

        self.use_fp16 = (
            False
            if self.running_config.use_fp16 == "none"
            else self.running_config.use_fp16
        )
        self.grad_scaler = ShardedGradScaler() if self.use_fp16 == "float16" else None

        self.writer.write("CheckpointManager is initialized...")
        self.checkpoint = CheckpointManager(self.config, self)
        self.is_resumed = self.checkpoint.load_state_dict()

        self.engine = build_engine(self.config, self)

        self.log_interval = self.running_config.log_interval
        self.eval_interval = self.running_config.evaluation_interval
        self.save_interval = self.running_config.checkpoint_interval
        self.iter_per_update = self.running_config.iter_per_update
        self.iou_type = (
            tuple(self.running_config.iou_type)
            if self.running_config.iou_type is not None
            else None
        )

        self.max_update = self.running_config.max_update
        self.max_epoch = self.running_config.max_epoch

        if self.max_epoch is not None and self.max_update is not None:
            raise ValueError("max_epoch and max_update are mutually exclusive!")

        batch_size = self.running_config.batch_size
        if self.dataloaders["train"] is not None:
            update_per_epoch = len(self.datasets["train"]) // batch_size

            if self.max_epoch is not None:
                self.max_update = self.max_epoch * update_per_epoch
            self.eval_interval = int(self.eval_interval * update_per_epoch)
            self.save_interval = int(self.save_interval * update_per_epoch)
        else:
            self.max_update = 0

        self.meters = {
            "train": Meter(),
            "val": Meter(),
            "test": Meter(),
        }
        self.timers = {"train": Timer(), "val": Timer(), "test": Timer()}

        self.eval_iteration = 0
        if self.datasets["val"] is not None:
            self.eval_iteration = len(self.dataloaders["val"])
        self.not_debug = self.running_config.logger_level != "debug"

        self.tb_writer = None
        if self.running_config.tensorboard:
            tb_log_folder = os.path.join(self.writer.log_folder, "tensorboard")

            if self.running_config.tensorboard_logdir:
                tb_log_folder = self.running_config.tensorboard_logdir

            if is_master() and not os.path.exists(tb_log_folder):
                os.makedirs(tb_log_folder)
            self.tb_writer = TensorboardLogger(tb_log_folder)

    def train(self):
        if "train" not in self.run_type:
            self.inference()
            return

        self.model.train()
        self.profile("Setup Time")

        torch.autograd.set_detect_anomaly(False)
        self.writer.write("Starting training...")

        if self.is_resumed:
            self.writer.write(f"Resuming training at {self.current_update}...")
            self.lr_scheduler.step_epoch(self.current_epoch)

        while self.current_update < self.max_update:
            self.current_epoch += 1
            self.engine.train_epoch()
        self.finalize()

    def _sync_losses_and_metrics(self, split, output):
        split_batch_size = (
            get_batch_size(self.running_config.batch_size) // self.iter_per_update
        )
        losses = output["losses_stat"]
        metrics = output["metrics"]

        reduced_losses = reduce_dict(losses, self.dp_mesh)
        reduced_metrics = reduce_dict(metrics, self.dp_mesh)

        update_dict = {}
        update_dict.update(reduced_losses)
        update_dict.update(reduced_metrics)

        self.meters[split].update(update_dict, split_batch_size)
