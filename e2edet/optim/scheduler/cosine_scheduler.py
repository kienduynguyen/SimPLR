import math

from e2edet.optim.scheduler import register_scheduler, BaseScheduler


@register_scheduler("cosine_annealing")
class CosineAnnealingScheduler(BaseScheduler):
    def __init__(self, config, optimizer):
        eta_min = config["eta_min"]
        self.T_max = config["T_max"]
        self.use_warmup = config["use_warmup"]
        self.warmup_iterations = config["warmup_iterations"] if self.use_warmup else 0
        self.warmup_factor = config["warmup_factor"]

        base_lrs = list(map(lambda group: group["lr"], optimizer.param_groups))
        max_base_lr = max(base_lrs)
        self.eta_mins = [lr / max_base_lr * eta_min for lr in base_lrs]
        if self.use_warmup and self.warmup_iterations > 0:
            self.T_max -= self.warmup_iterations

        super(CosineAnnealingScheduler, self).__init__(config, optimizer)

    def __repr__(self):
        format_string = self.__class__.__name__ + " (\n"
        for key in (
            "last_iter",
            "last_epoch",
            "use_warmup",
            "T_max",
            "eta_mins",
            "warmup_iterations",
            "warmup_factor",
        ):
            format_string += f"    {key}: {getattr(self, key)}\n"
        format_string += ")"
        return format_string

    def get_iter_lr(self):
        if self.last_iter <= self.warmup_iterations and self.use_warmup is True:
            alpha = float(self.last_iter) / float(self.warmup_iterations)
            lr_ratio = self.warmup_factor * (1.0 - alpha) + alpha

            return [base_lr * lr_ratio for base_lr in self.base_lrs]
        else:
            return [
                self.eta_mins[i]
                + (base_lr - self.eta_mins[i])
                * (
                    1
                    + math.cos(
                        math.pi * (self.last_iter - self.warmup_iterations) / self.T_max
                    )
                )
                / 2
                for i, base_lr in enumerate(self.base_lrs)
            ]
