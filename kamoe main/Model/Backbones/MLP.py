import torch
import torch.nn as nn
from typing import List, Optional
from ..utils.activation_functions import get_activation


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with flexible layer dimensions, activations,
    LayerNorm, and Dropout.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        layers_num: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        activation: str = 'relu',
        final_activation: Optional[str] = None,
        use_layernorm: bool = False
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim

        for i in range(layers_num - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        if final_activation is not None:
            layers.append(get_activation(final_activation))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
