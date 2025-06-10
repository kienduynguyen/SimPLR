import torch
import torch.distributed as dist

from e2edet.utils.general import filter_grads, get_memory_stats
from e2edet.module.parallelize import clip_grad_norm_


class BaseEngine:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = self.trainer.model
        self.optimizer = self.trainer.optimizer
        self.lr_scheduler = self.trainer.lr_scheduler
        self.dataloaders = self.trainer.dataloaders
        self.datasets = self.trainer.datasets
        self.params = filter_grads(self.model.parameters())
        self.num_skip = 0

    @torch.no_grad()
    def evaluate(self, split):
        raise NotImplementedError

    @property
    def current_epoch(self):
        current_update = self.trainer.current_update
        batch_size = self.trainer.running_config.batch_size
        if self.datasets["train"] is not None:
            update_per_epoch = len(self.datasets["train"]) // batch_size
        else:
            update_per_epoch = 1

        return (current_update + update_per_epoch - 1) // update_per_epoch

    def train_epoch(self):
        raise NotImplementedError

    def _compute_loss(self, output, target):
        raise NotImplementedError

    def _forward(self, batch, **kwargs):
        self.trainer.profile("Batch prepare time")

        sample, target = batch

        if self.trainer.use_fp16:
            assert self.trainer.use_fp16 in ("float16", "bfloat16")
            dtype = (
                torch.bfloat16 if self.trainer.use_fp16 == "bfloat16" else torch.float16
            )
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = self.model(sample, target)
                output = self._compute_loss(output, target, **kwargs)
        else:
            output = self.model(sample, target)
            output = self._compute_loss(output, target, **kwargs)
        self.trainer.profile("Forward time")

        return output, target

    def _backward(self, output):
        loss = output["losses"]

        if self.trainer.grad_scaler is not None:
            self.trainer.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        self.trainer.profile("Backward time")

    def _step(self, current_update):
        max_norm = self.trainer.running_config.max_norm
        if self.trainer.grad_scaler is not None:
            self.trainer.grad_scaler.unscale_(self.optimizer)
            self.trainer.profile("Unscale time")

        norm = clip_grad_norm_(self.params, max_norm)
        found_inf = torch.tensor([0], device=self.trainer.device)
        if torch.isnan(norm) or torch.isinf(norm):
            found_inf.fill_(1)
        dist.all_reduce(found_inf, group=dist.group.WORLD)
        self.trainer.profile("Clip grad time")

        if self.trainer.grad_scaler is not None:
            self.trainer.grad_scaler.step(self.optimizer)
            self.trainer.profile("Step time")
            self.trainer.grad_scaler.update()
            self.trainer.profile("Update time")
        else:
            self.optimizer.step()
            self.trainer.profile("Step time")

        if found_inf.item() > 0:
            self.num_skip += 1
            if self.num_skip >= 100:
                raise RuntimeError("Skipping iteration for more than 100 steps...")

            return current_update
        else:
            self.num_skip = 0

        if self.trainer.tb_writer is not None:
            self.trainer.tb_writer.add_scalars({"total_norm": norm}, current_update)

        current_update += 1

        return current_update

    @torch.no_grad()
    def _update_info(self, split, current_update):
        if split == "train":
            log_interval = self.trainer.log_interval
        else:
            log_interval = 1

        if current_update % log_interval == 0:
            stats = {}

            if "cuda" in str(self.trainer.device):
                stats.update(get_memory_stats(self.trainer.device))

            ups = log_interval / self.trainer.timers[split].unix_time_since_start()

            if split == "train":
                stats.update(
                    {
                        "epoch": self.current_epoch,
                        "data_epoch": self.trainer.current_epoch,
                        "update": current_update,
                        "max_update": self.trainer.max_update,
                        "lr": [
                            param_group["lr"]
                            for param_group in self.optimizer.param_groups
                        ],
                        "ups": "{:.2f}".format(ups),
                        "time": self.trainer.timers[split].get_time_since_start(),
                        "time_since_start": self.trainer.total_timer.get_time_since_start(),
                        "eta": self.trainer._calculate_time_left(),
                    }
                )
            else:
                stats.update(
                    {
                        "update": current_update,
                        "ups": "{:.2f}".format(ups),
                        "time": self.trainer.timers[split].get_time_since_start(),
                        "time_since_start": self.trainer.total_timer.get_time_since_start(),
                    }
                )
            self.trainer._print_log(split, stats)
            self.trainer._update_tensorboard(split)
            self.trainer.timers[split].reset()
        self.trainer.profile("Update info time")
