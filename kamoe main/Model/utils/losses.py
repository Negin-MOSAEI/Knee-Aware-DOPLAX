import torch
import torch.nn as nn


def mape_loss_fn(outputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean Absolute Percentage Error loss."""
    return torch.mean(torch.abs((targets - outputs) / (targets + eps)))


class KneeAwareLoss(nn.Module):
    """
    Knee-Aware Loss function:
    Applies higher penalty/weight to degradation errors after the knee point (or weighted by knee distance).
    
    L = L_pre + alpha * L_post
    where alpha > 1 ensures the accelerated degradation phase is tightly modeled.
    """
    def __init__(self, post_knee_weight: float = 2.0, base_loss: str = 'mse'):
        super(KneeAwareLoss, self).__init__()
        self.post_knee_weight = post_knee_weight
        if base_loss == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif base_loss == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif base_loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            self.criterion = nn.MSELoss(reduction='none')

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        is_post_knee: torch.Tensor = None
    ) -> torch.Tensor:
        loss = self.criterion(pred, target)
        if is_post_knee is not None:
            # is_post_knee is 1 for post-knee cycles and 0 for pre-knee cycles
            weights = 1.0 + (self.post_knee_weight - 1.0) * is_post_knee
            loss = loss * weights
        return loss.mean()
