import math
import random
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

from ..Backbones.MLP import MLP
from ..utils.losses import mape_loss_fn
try:
    from configs import LAXConfig, get_lax_config
except ImportError:
    from ...configs import LAXConfig, get_lax_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class theta_inverser(nn.Module):
    """Computes theta(t) = 1 / (t + 1e-8)"""
    def __init__(self):
        super(theta_inverser, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (x + 1e-8)


class LpDistance(nn.Module):
    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(x - y)
        dist = torch.pow(torch.sum(diff ** self.p, dim=-1, keepdim=True), 1.0 / self.p)
        return dist


class LearnableDistance(nn.Module):
    def __init__(self, dim: int = 16, embed_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.L = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ex = self.encoder(x)
        ey = self.encoder(y)
        diff = ex - ey
        W = self.L.T @ self.L
        dist2 = torch.sum(diff @ W * diff, dim=-1, keepdim=True)
        return torch.sqrt(dist2 + 1e-6)


class SumProductNetwork(nn.Module):
    """
    Symmetric distance modeling block: d(x, y) = f(x+y, x*y)
    """
    def __init__(self, d_in: int, h_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 64)
        self.ln1 = nn.LayerNorm(64)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(64, h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sum_xy = x + y
        prod_xy = x * y
        z = torch.cat([sum_xy, prod_xy], dim=-1)
        z = self.act1(self.ln1(self.fc1(z)))
        z = self.act2(self.ln2(self.fc2(z)))
        return z


class Integrator(nn.Module):
    """
    Numerical trapezoidal integrator for evaluating Hamilton-Jacobi action functional.
    """
    def __init__(self, phi_network: nn.Module, n_steps: int = 200):
        super().__init__()
        self.phi = phi_network
        self.n_steps = n_steps

    def forward(self, h_in: torch.Tensor, t_u: torch.Tensor) -> torch.Tensor:
        if t_u.dim() > 1:
            t_u = t_u.squeeze(-1)
        t_min = torch.zeros_like(t_u)
        t_max = t_u
        t_steps = torch.linspace(0, 1, self.n_steps).unsqueeze(0).to(t_u.device)
        t_integral_points = t_min.unsqueeze(-1) + (t_max.unsqueeze(-1) - t_min.unsqueeze(-1)) * t_steps
        h_in_expanded = h_in.unsqueeze(1).expand(-1, self.n_steps, -1)

        phi_input = torch.cat([h_in_expanded, t_integral_points.unsqueeze(-1)], dim=-1)
        phi_integral_out = self.phi(phi_input.view(-1, phi_input.shape[-1]))
        phi_integral_out = phi_integral_out.view(h_in.shape[0], self.n_steps, -1)

        integral_sum = torch.sum(phi_integral_out[:, 1:-1, :], dim=1) + \
                       0.5 * (phi_integral_out[:, 0, :] + phi_integral_out[:, -1, :])
        step_size = (t_max - t_min) / max(1, self.n_steps - 1)
        return step_size.unsqueeze(-1) * integral_sum


class OptimizationNetwork(nn.Module):
    """
    LAX Network: Hopf-Lax solver for Hamilton-Jacobi equations with learned Hamiltonian.
    Solves: u(x, t) = min_y [ g(y) + t * H*((x - y)/t) ]
    Parameters match the optimal settings from the DOPLAX project.
    """
    def __init__(
        self,
        config: Optional[Union[LAXConfig, str]] = None,
        x_dim: int = 16,
        y_dim: int = 16,
        h_dim: int = 49,
        center_block: str = 'PhI',
        distance_block: str = 'SumProductNetwork',
        time_block: str = 'theta',
        x_sts: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        super().__init__()

        if isinstance(config, str):
            config = get_lax_config(config)

        if isinstance(config, LAXConfig):
            self.dataset = config.dataset
            self.x_dim = config.x_dim
            self.y_dim = config.y_dim
            self.h_dim = config.h_dim
            self.distance_block = config.distance_block
            self.center_block = config.center_block
            self.time_block = config.time_block
            self.s_block = config.s_block
            self.lr_net = config.lr_net
            self.lr_y = config.lr_y
            self.betha_LAX = config.betha_LAX
            self.theta_LAX = config.theta_LAX
            self.dual_LAX = config.dual_LAX
            self.inside_h_star_layers = config.inside_h_star_layers
            self.inside_phi_layers = config.inside_phi_layers
            self.inside_g_layers = config.inside_g_layers
            self.inside_S_MLP_layers = config.inside_S_MLP_layers
            self.inside_distance_block_MLP_layers = config.inside_distance_block_MLP_layers
            self.inside_theta_layers = config.inside_theta_layers
            self.zeta_LAX = getattr(config, 'zeta_LAX', 1.0)
            self.kata_LAX = getattr(config, 'kata_LAX', 0.0)
        else:
            self.dataset = "XJTU"
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.h_dim = h_dim
            self.distance_block = distance_block
            self.center_block = center_block
            self.time_block = time_block
            self.s_block = "MLP"
            self.lr_net = 0.040
            self.lr_y = 0.080
            self.betha_LAX = 0.0256
            self.theta_LAX = 0.0
            self.dual_LAX = 0.0
            self.zeta_LAX = 1.0
            self.kata_LAX = 0.0
            self.inside_h_star_layers = [64, 32]
            self.inside_phi_layers = [64, 32]
            self.inside_g_layers = [16]
            self.inside_S_MLP_layers = [32, 16, 8]
            self.inside_distance_block_MLP_layers = [64]
            self.inside_theta_layers = [32]

        self.d_in = self.x_dim + self.y_dim

        if x_sts is not None:
            self.register_buffer('X_mean', x_sts[0].to(torch.float32))
            self.register_buffer('X_std', x_sts[1].to(torch.float32))
        else:
            self.register_buffer('X_mean', torch.zeros(self.x_dim))
            self.register_buffer('X_std', torch.ones(self.x_dim))

        # Dynamic parameter y for minimization
        self.y_param = nn.Parameter(torch.zeros(1, self.y_dim), requires_grad=True)

        # 1. Distance block
        if self.distance_block == 'SumProductNetwork':
            self.d = SumProductNetwork(d_in=self.d_in, h_dim=self.h_dim)
        elif self.distance_block == 'Mahalanobis':
            self.d = LearnableDistance(dim=self.x_dim, embed_dim=self.h_dim)
        elif self.distance_block == 'Lp_norm':
            self.d = LpDistance(p=2.0)
        else:
            self.d = nn.Sequential(
                nn.Linear(self.d_in, self.inside_distance_block_MLP_layers[0]),
                nn.SiLU(),
                nn.Linear(self.inside_distance_block_MLP_layers[0], self.h_dim),
                nn.ReLU()
            )

        # 2. Time block
        if self.time_block == '1/t':
            self.theta = theta_inverser()
        else:
            self.theta = nn.Sequential(
                nn.Linear(1, self.inside_theta_layers[0]),
                nn.SiLU(),
                nn.Linear(self.inside_theta_layers[0], self.h_dim)
            )

        # 3. Hamiltonian H*
        self.H_star = nn.Sequential(
            nn.Linear(self.h_dim, self.inside_h_star_layers[0]),
            nn.SiLU(),
            nn.Linear(self.inside_h_star_layers[0], self.inside_h_star_layers[1]),
            nn.SiLU(),
            nn.Linear(self.inside_h_star_layers[1], 1)
        )

        # 4. Phi Network & Integrator
        self.phi = nn.Sequential(
            nn.Linear(self.h_dim + 1, self.inside_phi_layers[0]),
            nn.ReLU(),
            nn.Linear(self.inside_phi_layers[0], self.inside_phi_layers[1]),
            nn.ReLU(),
            nn.Linear(self.inside_phi_layers[1], 1)
        )
        self.integrator = Integrator(self.phi, n_steps=200)

        # 5. Initial condition g(y)
        self.g = nn.Sequential(
            nn.Linear(self.y_dim, self.inside_g_layers[0]),
            nn.SiLU(),
            nn.Linear(self.inside_g_layers[0], 1)
        )

        # 6. Final solver combination network s(g, t H*)
        self.s_MLP = nn.Sequential(
            nn.Linear(2, self.inside_S_MLP_layers[0]),
            nn.SiLU(),
            nn.Linear(self.inside_S_MLP_layers[0], self.inside_S_MLP_layers[1]),
            nn.SiLU(),
            nn.Linear(self.inside_S_MLP_layers[1], self.inside_S_MLP_layers[2]),
            nn.SiLU(),
            nn.Linear(self.inside_S_MLP_layers[2], 1)
        )

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, epoch: Optional[int] = None) -> torch.Tensor:
        """
        :param x: Feature tensor of shape [N, x_dim] (without time) or [N, x_dim+1] (with time in last column).
        :param t: Optional time/cycle tensor of shape [N, 1] or [N].
        :return: Estimated SOH capacity u(x, t) of shape [N, 1].
        """
        if t is None:
            t = x[:, -1:]
            x_feat = x[:, :self.x_dim]
        else:
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            x_feat = x if x.shape[1] == self.x_dim else x[:, :self.x_dim]

        batch_size = x_feat.shape[0]
        y_expanded = self.y_param.expand(batch_size, -1)

        # Compute distance d(x, y)
        d_val = self.d(x_feat, y_expanded)

        # Compute theta(t)
        theta_val = self.theta(t)
        if theta_val.shape[-1] != d_val.shape[-1]:
            theta_val = theta_val.expand(-1, d_val.shape[-1])

        # State argument z = d(x, y) * theta(t)
        z = d_val * theta_val

        # Center block evaluation (PhI vs H*)
        if self.center_block == 'PhI':
            h_val = self.integrator(z, t)
        else:
            h_val = t * self.H_star(z)

        # Initial condition g(y)
        g_val = self.g(y_expanded)

        # Hopf-Lax formula combination: s(g(y), Action)
        u_out = self.s_MLP(torch.cat([g_val, h_val], dim=-1))
        return u_out

    def compute_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes LAX loss: MSE data loss + Monotonicity loss + optional MAPE/MAE losses.
        """
        u1 = self.forward(x1)
        u2 = self.forward(x2)

        loss_mse = 0.5 * nn.functional.mse_loss(u1, y1) + 0.5 * nn.functional.mse_loss(u2, y2)
        loss_mae = 0.5 * nn.functional.l1_loss(u1, y1) + 0.5 * nn.functional.l1_loss(u2, y2)
        loss_mape = 0.5 * mape_loss_fn(u1, y1) + 0.5 * mape_loss_fn(u2, y2)
        loss_mono = torch.relu(torch.mul(u2 - u1, y1 - y2)).mean()

        total_loss = (
            self.zeta_LAX * loss_mse +
            self.betha_LAX * loss_mono +
            self.theta_LAX * loss_mape +
            self.kata_LAX * loss_mae
        )

        return {
            'total_loss': total_loss,
            'loss_mse': loss_mse,
            'loss_mae': loss_mae,
            'loss_mape': loss_mape,
            'loss_mono': loss_mono,
            'u1': u1,
            'u2': u2
        }

