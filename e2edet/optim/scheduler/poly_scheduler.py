import math

from e2edet.optim.scheduler import register_scheduler, BaseScheduler


@register_scheduler("poly")
class PolyScheduler(BaseScheduler):
    def __init__(self, config, optimizer):
        self.use_warmup = config["use_warmup"]
        self.max_iters = config["max_iters"]
        self.power = config["power"]
        self.constant_ending = config["constant_ending"]
        self.warmup_iterations = config.get("warmup_iterations", 0)
        self.warmup_factor = config.get("warmup_factor", 1)
        super().__init__(config, optimizer)

    def __repr__(self):
        format_string = self.__class__.__name__ + " (\n"
        for key in (
            "last_iter",
            "last_epoch",
            "use_warmup",
            "max_iters",
            "power",
            "constant_ending",
            "warmup_iterations",
            "warmup_factor",
        ):
            format_string += f"    {key}: {getattr(self, key)}\n"
        format_string += ")"
        return format_string

    def get_iter_lr(self):
        if self.last_iter >= self.warmup_iterations or self.use_warmup is False:
            warmup_factor = 1.0
        else:
            alpha = float(self.last_iter) / float(self.warmup_iterations)
            warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha

        if self.constant_ending > 0 and warmup_factor == 1.0:
            # Constant ending lr.
            if (
                math.pow((1.0 - self.last_iter / self.max_iters), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        return [
            base_lr
            * warmup_factor
            * math.pow((1.0 - self.last_iter / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]
