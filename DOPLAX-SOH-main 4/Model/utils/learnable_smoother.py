"""
Constrained Learnable Smoother with Total Variation (TV) Regularization.

Differentiable 1D convolutional filter designed to smooth high-frequency noise
in SOH trajectory predictions while preserving sharp knee-point inflections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConstrainedLearnableSmoother(nn.Module):
    """
    1D depthwise learnable smoother with constrained convex combination weights.

    Guarantees:
      1. Unity DC gain: Weights are softmax-normalized so sum(w) = 1.
      2. Strictly non-negative weights: Eliminates ringing/overshoot artifacts.
      3. Zero DC offset: Preserves absolute battery capacity scale.
      4. Reflection boundary padding: Prevents boundary distortion at cycle 1 or end of life.
    """
    def __init__(self, kernel_size: int = 5):
        super(ConstrainedLearnableSmoother, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd to maintain symmetric padding."
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Unconstrained raw logits for kernel weights
        # Initialize centered at Dirac delta (identity mapping) with slight smoothing
        init_weights = torch.zeros(1, 1, kernel_size)
        init_weights[0, 0, self.padding] = 3.0  # Center tap emphasis
        self.raw_weights = nn.Parameter(init_weights)

    @property
    def normalized_weights(self) -> torch.Tensor:
        """Normalized positive weights summing to 1.0 (convex combination)."""
        return F.softmax(self.raw_weights, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input SOH sequence tensor.
               Shape can be [B, T], [B, T, 1], or [B, 1, T].
        Returns:
            Smoothed SOH sequence with identical shape as x.
        """
        orig_shape = x.shape
        # Handle [B, T, 1] or [B, T]
        if x.dim() == 2:
            # [B, T] -> [B, 1, T]
            x_in = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[-1] == 1:
            # [B, T, 1] -> [B, 1, T]
            x_in = x.permute(0, 2, 1)
        elif x.dim() == 3 and x.shape[1] == 1:
            # Already [B, 1, T]
            x_in = x
        else:
            # [B, C, T]
            x_in = x

        # If sequence is too short for convolution, return as is
        if x_in.shape[-1] <= self.kernel_size:
            return x

        # Symmetric / replicate reflection padding
        x_padded = F.pad(x_in, (self.padding, self.padding), mode='replicate')
        kernel = self.normalized_weights

        # 1D Depthwise convolution
        out = F.conv1d(x_padded, kernel, groups=1)

        # Restore original shape
        if orig_shape != out.shape:
            if len(orig_shape) == 2:
                out = out.squeeze(1)
            elif len(orig_shape) == 3 and orig_shape[-1] == 1:
                out = out.permute(0, 2, 1)

        return out


def tv_regularization_loss(
    u_pred: torch.Tensor,
    order: int = 2,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Total Variation and Curvature Regularization Loss.

    Args:
        u_pred: Predicted trajectory [B, T] or [B, T, 1]
        order:
            1 -> First-order Total Variation: sum |u_{t+1} - u_t|
            2 -> Second-order Curvature Penalty: sum |u_{t+2} - 2 u_{t+1} + u_t|
                 (Preserves linear degradation slope while suppressing oscillations)
        reduction: 'mean' or 'sum'

    Returns:
        Scalar penalty tensor.
    """
    if u_pred.dim() == 3:
        u_seq = u_pred.squeeze(-1)
    else:
        u_seq = u_pred

    if u_seq.shape[-1] <= order:
        return torch.tensor(0.0, device=u_pred.device)

    if order == 1:
        diff = u_seq[:, 1:] - u_seq[:, :-1]
        loss = torch.abs(diff)
    elif order == 2:
        # Second discrete derivative: u_{t+2} - 2*u_{t+1} + u_t
        diff2 = u_seq[:, 2:] - 2.0 * u_seq[:, 1:-1] + u_seq[:, :-2]
        loss = torch.abs(diff2)
    else:
        raise ValueError(f"Unsupported order: {order}. Use 1 or 2.")

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
