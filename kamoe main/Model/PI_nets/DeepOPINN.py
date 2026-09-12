import os
import math
from typing import Optional, Dict, Any, Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn

from ..Backbones.DeepONet import DeepONet
from ..Backbones.MLP import MLP
from ..Auxiliary_nets.Solution_u import Solution_u
from ..utils.losses import KneeAwareLoss, mape_loss_fn
from ..utils.util import AverageMeter, eval_metrix
try:
    from configs import DeepOPINNConfig, get_deepopinn_config
except ImportError:
    from ...configs import DeepOPINNConfig, get_deepopinn_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepOPINN(nn.Module):
    """
    DeepOPINN: Deep Operator Physics-Informed Neural Network for Battery SOH estimation.
    Unifies operator learning (DeepONet) with PDE residual physics loss and SOH solution prediction.
    Parameters match the optimal settings from the DOPLAX project.
    """
    def __init__(
        self,
        config: Optional[Union[DeepOPINNConfig, str]] = None,
        input_dim: int = 17,
        hidden_dim: int = 80,
        layers_num: int = 5,
        dropout: float = 0.126,
        dim_x: int = 1,
        pde_weight: float = 0.118,
        monotone_weight: float = 0.016,
    ):
        super(DeepOPINN, self).__init__()

        # If a dataset string or DeepOPINNConfig is provided, load preset parameters
        if isinstance(config, str):
            config = get_deepopinn_config(config)

        if isinstance(config, DeepOPINNConfig):
            self.dataset = config.dataset
            self.input_dim = config.input_dim
            self.hidden_dim = config.hidden_dim
            self.layers_num = config.layers_num
            self.dropout = config.dropout
            self.dim_x = config.dim_x
            self.pde_weight = config.alpha
            self.monotone_weight = config.beta
            self.lr = config.lr
            self.warmup_lr = config.warmup_lr
            self.final_lr = config.final_lr
            self.lr_F = config.lr_F
            self.warmup_epochs = config.warmup_epochs
        else:
            self.dataset = "XJTU"
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.layers_num = layers_num
            self.dropout = dropout
            self.dim_x = dim_x
            self.pde_weight = pde_weight
            self.monotone_weight = monotone_weight
            self.lr = 0.00625
            self.warmup_lr = 0.00627
            self.final_lr = 1.6e-6
            self.lr_F = 0.00882
            self.warmup_epochs = 30

        # 1. Feature Extractor (DeepONet branch + trunk)
        branch_sizes = [self.input_dim] + [self.hidden_dim] * (self.layers_num - 1)
        trunk_sizes = [self.dim_x] + [self.hidden_dim] * (self.layers_num - 1)
        self.extractor = DeepONet(
            layer_sizes_branch=branch_sizes,
            layer_sizes_trunk=trunk_sizes,
            activation="relu",
            kernel_initializer="Glorot normal"
        )

        # 2. SOH Solution Network u(x)
        # Input to solution_u is (num_coords + 1) metadata features
        self.num_coords = self.input_dim - 1
        self.solution_u = Solution_u(
            input_dim=self.num_coords + 1,
            output_dim=32,
            layers_num=self.layers_num,
            hidden_dim=60,
            dropout=self.dropout
        )

        # 3. Dynamical PDE residual network F(x1, x2, u1, u2, ...)
        f_in_dim = (self.input_dim - 1) * 2 + 3
        hidden_dim_F = self.hidden_dim if self.hidden_dim <= 20 else self.hidden_dim * 2
        self.dynamical_F = MLP(
            input_dim=f_in_dim,
            output_dim=1,
            layers_num=self.layers_num,
            hidden_dim=hidden_dim_F,
            dropout=self.dropout
        )

        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts operator-level features using DeepONet Cartesian product.
        """
        metadata = x[:, -1].unsqueeze(1)
        coords = torch.linspace(0, 1, self.num_coords, device=x.device).reshape(-1, 1)
        features = self.extractor((x, coords))  # [B, num_coords]
        return torch.cat([features, metadata], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: Input battery feature tensor [B, input_dim].
        :return: (u_encoded, u_pred)
        """
        xt = self.extract_features(x)
        u_encoded, u = self.solution_u(xt)
        return u_encoded, u

    def compute_pde_residual(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        u1: torch.Tensor,
        u2: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes PDE residual along consecutive transition steps (x1 -> x2).
        """
        delta_x = x2[:, :-1] - x1[:, :-1]
        delta_u = u2 - u1
        state_input = torch.cat([x1[:, :-1], delta_x, u1, u2, delta_u], dim=-1)
        f_pred = self.dynamical_F(state_input)
        return f_pred

    def compute_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
        kpd1: Optional[torch.Tensor] = None,
        is_post_knee1: Optional[torch.Tensor] = None,
        is_post_knee2: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes full training loss = Data Loss + Knee-Aware PDE Loss + Monotonicity Loss.
        PDE loss is weighted by max(0, kpd1) across each cycle transition.
        """
        _, u1 = self.forward(x1)
        _, u2 = self.forward(x2)

        # 1. Data Loss
        loss_data = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)

        # 2. Knee-Aware PDE Residual Loss: (f_residual^2) * max(0, kpd)
        f_residual = self.compute_pde_residual(x1, x2, u1, u2)
        if kpd1 is not None:
            if kpd1.dim() == 1:
                kpd1 = kpd1.unsqueeze(-1)
            pde_weight_cycle = torch.clamp(kpd1, min=0.0)
            loss_pde = torch.mean((f_residual ** 2) * pde_weight_cycle)
        else:
            loss_pde = torch.mean(f_residual ** 2)

        # 3. Monotonicity Loss (Capacity decreases across degradation cycles)
        loss_mono = self.relu(torch.mul(u2 - u1, y1 - y2)).mean()

        total_loss = loss_data + self.pde_weight * loss_pde + self.monotone_weight * loss_mono

        return {
            'total_loss': total_loss,
            'loss_data': loss_data,
            'loss_pde': loss_pde,
            'loss_mono': loss_mono,
            'u1': u1,
            'u2': u2
        }

