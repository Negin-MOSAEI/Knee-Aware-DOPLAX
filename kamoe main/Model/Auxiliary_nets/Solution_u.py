import torch
import torch.nn as nn
from ..utils.activation_functions import Sin
from ..Backbones.MLP import MLP


class Predictor(nn.Module):
    def __init__(self, input_dim: int = 32, dropout: float = 0.2):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, input_dim),
            Sin(),
            nn.Linear(input_dim, 1)
        )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Solution_u(nn.Module):
    """
    Encoder + Predictor network estimating SOH solution u(x, t).
    """
    def __init__(
        self,
        input_dim: int = 17,
        output_dim: int = 32,
        layers_num: int = 3,
        hidden_dim: int = 60,
        dropout: float = 0.2,
        LAX_model: nn.Module = None
    ):
        super(Solution_u, self).__init__()
        self.LAX_model = LAX_model
        self.encoder = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            layers_num=layers_num,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        if self.LAX_model is None:
            self.predictor = Predictor(input_dim=output_dim, dropout=dropout)
        else:
            self.predictor = self.LAX_model
        self._init_weights()

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, epoch: int = None):
        u_encoded = self.encoder(x)
        if self.LAX_model is None:
            u = self.predictor(u_encoded)
        else:
            t = x[:, -1]
            u = self.predictor(x=u_encoded, t=t, epoch=epoch)
        return u_encoded, u

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
