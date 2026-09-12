import torch
import torch.nn as nn
from typing import List, Optional


class Chomp1d(nn.Module):
    """
    Removes the padding on the right to ensure causal convolution (no future leakage).
    """
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    A single Residual Block in the Temporal Convolutional Network (TCN).
    Consists of two dilated causal 1D convs, weight normalization, ReLU, dropout, and residual skip.
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence modeling of battery trajectories.
    Predicts knee point distance or knee onset sequentially without shuffling.
    """
    def __init__(
        self,
        num_inputs: int = 17,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_dim: int = 1
    ):
        super(TemporalConvNet, self).__init__()
        if num_channels is None:
            num_channels = [32, 64, 128, 64, 32]

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size, padding=padding,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input sequence tensor.
                  Can be shape [L, D] (single battery) or [B, L, D] (batch of batteries).
        :return: Output knee distance predictions of shape [L, output_dim] or [B, L, output_dim].
        """
        is_2d = (x.dim() == 2)
        if is_2d:
            x = x.unsqueeze(0)  # [1, L, D]

        # Conv1d expects [B, Channels, Length]
        x_conv = x.transpose(1, 2)
        feat = self.network(x_conv)  # [B, Channels_out, Length]
        feat = feat.transpose(1, 2)  # [B, Length, Channels_out]
        out = self.head(feat)        # [B, Length, output_dim]

        if is_2d:
            return out.squeeze(0)    # [L, output_dim]
        return out
