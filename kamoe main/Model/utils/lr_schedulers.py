import math
import torch


class LR_Scheduler:
    """
    Learning Rate Scheduler with linear warmup and cosine annealing decay.
    """
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr=0.0):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, warmup_epochs)
        self.warmup_lr = warmup_lr
        self.num_epochs = max(1, num_epochs)
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.current_epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs and self.warmup_epochs > 0:
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(1, (self.num_epochs - self.warmup_epochs))
            progress = min(1.0, max(0.0, progress))
            lr = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
