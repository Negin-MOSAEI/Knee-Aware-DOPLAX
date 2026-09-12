import torch
import torch.nn as nn

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

def get_activation(act_name: str) -> nn.Module:
    name = act_name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sin':
        return Sin()
    elif name == 'silu':
        return nn.SiLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'none' or name == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {act_name}")
