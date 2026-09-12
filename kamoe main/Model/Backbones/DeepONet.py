import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet) with Branch Network and Trunk Network.
    G(u)(y) = dot_product(Branch(u), Trunk(y)) + bias
    """
    def __init__(
        self,
        layer_sizes_branch: List[int],
        layer_sizes_trunk: List[int],
        activation: str = "relu",
        kernel_initializer: str = "Glorot normal"
    ):
        super().__init__()
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()
        self.branch = self._build_network(layer_sizes_branch)
        self.trunk = self._build_network(layer_sizes_trunk)
        self.b0 = nn.Parameter(torch.tensor(0.0))
        self._init_weights(kernel_initializer)

    def _build_network(self, layer_sizes: List[int]) -> nn.Sequential:
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(self.activation)
        return nn.Sequential(*layers)

    def _init_weights(self, initializer: str):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if initializer == "Glorot normal":
                    nn.init.xavier_normal_(m.weight)
                elif initializer == "He normal":
                    nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        func, spatial = inputs
        branch_out = self.branch(func)      # [B, p]
        trunk_out = self.trunk(spatial)    # [N, p]

        # Cartesian product dot-product
        output = torch.einsum('bi,ci->bc', branch_out, trunk_out) + self.b0
        return output
