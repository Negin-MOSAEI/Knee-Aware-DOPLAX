import torch
import torch.nn as nn
from typing import List, Optional


class MLP_Bagging_NN(nn.Module):
    """
    Bagging MLP combining [u_pinn, u_lax] predictions.
    """
    def __init__(self, input_dim: int = 2, output_dim: int = 1, hidden_dim: Optional[List[int]] = None, dropout: float = 0.2):
        super(MLP_Bagging_NN, self).__init__()
        if hidden_dim is None:
            hidden_dim = [50]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = len(hidden_dim) + 1

        layers = []
        for i in range(self.layers_num):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim[i]))
                layers.append(nn.Tanh())
            elif i == self.layers_num - 1:
                layers.append(nn.Linear(hidden_dim[i - 1], output_dim))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
                layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
