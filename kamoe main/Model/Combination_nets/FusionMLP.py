import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List, Union

from ..Backbones.MLP import MLP
try:
    from configs import FusionMLPConfig, get_fusion_config
except ImportError:
    from ...configs import FusionMLPConfig, get_fusion_config


class FusionMLP(nn.Module):
    """
    Learned Fusion MLP combining DeepOPINN operator predictions,
    LAX Hopf-Lax solver predictions, and TCN knee-distance features.
    Matches the exact Bagging architecture from the previous project with optional knee-feature integration.
    """
    def __init__(
        self,
        config: Optional[Union[FusionMLPConfig, str]] = None,
        input_dim: int = 2,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.126,
        include_knee_feature: bool = True
    ):
        super(FusionMLP, self).__init__()

        if isinstance(config, str):
            config = get_fusion_config(config)

        if isinstance(config, FusionMLPConfig):
            self.dataset = config.dataset
            self.hidden_dims = config.hidden_dims
            self.dropout = config.dropout
            self.include_knee_feature = config.include_knee_feature
            self.bagging_lr = config.bagging_lr
            self.mono_bag = config.mono_bag
            self.epochs = config.epochs
            self.early_stop = config.early_stop
            # If include_knee_feature is True, input is [u_pinn, u_lax, knee_dist] -> dim = 3
            self.input_dim = 3 if self.include_knee_feature else 2
        else:
            self.dataset = "XJTU"
            self.include_knee_feature = include_knee_feature
            self.input_dim = input_dim + (1 if include_knee_feature and input_dim == 2 else 0)
            self.hidden_dims = hidden_dims if hidden_dims is not None else [10, 10]
            self.dropout = dropout
            self.bagging_lr = 0.020
            self.mono_bag = 0.40
            self.epochs = 2000
            self.early_stop = 80

        layers = []
        curr_dim = self.input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.Tanh())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            curr_dim = h

        layers.append(nn.Linear(curr_dim, 1))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        u_deepopinn: torch.Tensor,
        u_lax: torch.Tensor,
        knee_distance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        :param u_deepopinn: [N, 1] SOH prediction from DeepOPINN
        :param u_lax: [N, 1] SOH prediction from LAX
        :param knee_distance: Optional [N, 1] predicted knee distance from TCN
        :return: [N, 1] final fused SOH prediction
        """
        inputs = [u_deepopinn, u_lax]
        if self.include_knee_feature and knee_distance is not None:
            inputs.append(knee_distance)

        fused_input = torch.cat(inputs, dim=-1)
        return self.net(fused_input)

    def compute_loss(
        self,
        u_dop_1: torch.Tensor,
        u_lax_1: torch.Tensor,
        y1: torch.Tensor,
        u_dop_2: torch.Tensor,
        u_lax_2: torch.Tensor,
        y2: torch.Tensor,
        kpd_1: Optional[torch.Tensor] = None,
        kpd_2: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes FusionMLP loss: Data MSE loss + Monotonicity loss.
        """
        u_fused_1 = self.forward(u_dop_1, u_lax_1, kpd_1)
        u_fused_2 = self.forward(u_dop_2, u_lax_2, kpd_2)

        loss_data = 0.5 * nn.functional.mse_loss(u_fused_1, y1) + 0.5 * nn.functional.mse_loss(u_fused_2, y2)
        loss_mono = torch.relu(torch.mul(u_fused_2 - u_fused_1, y1 - y2)).mean()
        total_loss = loss_data + self.mono_bag * loss_mono

        return {
            'total_loss': total_loss,
            'loss_data': loss_data,
            'loss_mono': loss_mono,
            'u1': u_fused_1,
            'u2': u_fused_2
        }

