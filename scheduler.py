import math


class CosineScheduler:
    def __init__(
        self,
        lr_warmup_fraction: float,
        lr: float,
        final_lr_fraction: float,
        n_steps: int,
    ):
        self.lr_warmup_steps = n_steps * lr_warmup_fraction
        self.lr = lr
        self.n_steps = n_steps
        self.final_lr_fraction = final_lr_fraction

    def get_lr(self, step: int):
        if step < self.lr_warmup_steps:
            return self.lr * (step + 1) / self.lr_warmup_steps
        # cosine schedule that ends at final_lr_fraction * lr, then constant
        elif step < self.n_steps:
            return self.final_lr_fraction * self.lr + 0.5 * (
                1 - self.final_lr_fraction
            ) * self.lr * (
                1
                + math.cos(
                    math.pi
                    * (step - self.lr_warmup_steps)
                    / (self.final_lr_step - self.lr_warmup_steps)
                )
            )
        else:
            return self.lr * self.final_lr_fraction

    def set_lr(step, optimizer):
        new_lr = self.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
