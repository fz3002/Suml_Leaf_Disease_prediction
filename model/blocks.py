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
        self.in_channels: int = in_channels
        self.squeeze_channels: int = squeeze_channels
        self.expand1x1_channels: int = expand1x1_channels
        self.expand3x3_channels: int = expand3x3_channels

        self.squeeze: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=squeeze_channels,
            kernel_size=1
        )
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1: nn.Conv2d = nn.Conv2d(
            in_channels=squeeze_channels,
            out_channels=expand1x1_channels,
            kernel_size=1
        )
        self.expand3x3: nn.Conv2d = nn.Conv2d(
            in_channels=squeeze_channels,
            out_channels=expand3x3_channels,
            kernel_size=3,
            padding=1
        )
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.squeeze(x)
        x: torch.Tensor = self.squeeze_activation(x)
        out1: torch.Tensor = self.expand1x1(x)
        out2: torch.Tensor = self.expand3x3(x)
        x: torch.Tensor = torch.cat([out1, out2], dim=1)
        x: torch.Tensor = self.expand_activation(x)
        return x
