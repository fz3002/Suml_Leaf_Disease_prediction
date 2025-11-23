import torch
from torch import nn


class FireBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 squeeze_channels: int,
                 expand1x1_channels: int,
                 expand3x3_channels: int,
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.squeeze_channels = squeeze_channels
        self.expand1x1_channels = expand1x1_channels
        self.expand3x3_channels = expand3x3_channels

        self.squeeze = nn.Conv2d(
            in_channels,
            squeeze_channels,
            kernel_size=1
        )
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(
            squeeze_channels,
            expand1x1_channels,
            kernel_size=1
        )
        self.expand3x3 = nn.Conv2d(
            squeeze_channels,
            expand3x3_channels,
            kernel_size=3,
            padding=1
        )
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        x = self.squeeze_activation(x)
        out1 = self.expand1x1(x)
        out2 = self.expand3x3(x)
        x = torch.cat([out1, out2], dim=1)
        x = self.expand_activation(x)
        return x
