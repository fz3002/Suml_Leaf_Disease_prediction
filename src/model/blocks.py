import torch
from torch import nn


class FireBlock(nn.Module):
    """
    Fire module used in the SqueezeNet architecture.

    The FireBlock consists of two stages:
    1. Squeeze layer: a 1x1 convolution that reduces the number of input channels.
    2. Expand layer: a combination of 1x1 and 3x3 convolutions whose outputs are concatenated along the channel dimension.

    This design significantly reduces the number of parameters while maintaining representational power.
    """

    def __init__(self, in_channels: int, squeeze_channels: int, expand1x1_channels: int, expand3x3_channels: int) -> None:
        """
        FireBlock constructor.

        :param in_channels: Number of channels in the input feature map.
        :param squeeze_channels: Number of channels produced by the squeeze (1x1) convolution.
        :param expand1x1_channels: Number of channels produced by the expand 1x1 convolution.
        :param expand3x3_channels: Number of channels produced by the expand 3x3 convolution.
        """

        super().__init__()
        self.in_channels: int = in_channels
        self.squeeze_channels: int = squeeze_channels
        self.expand1x1_channels: int = expand1x1_channels
        self.expand3x3_channels: int = expand3x3_channels

        # Squeeze layer: reduces channel dimensionality
        self.squeeze: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=squeeze_channels,
            kernel_size=1
        )
        self.squeeze_activation = nn.ReLU(inplace=True)

        # Expand layer: two parallel convolutions
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
        """
        Forward pass of the FireBlock.

        :param x: Input tensor (image)
        :return: Output tensor
        """

        # Squeeze stage
        x: torch.Tensor = self.squeeze(x)
        x: torch.Tensor = self.squeeze_activation(x)

        # Expand stage, parallel paths
        out1: torch.Tensor = self.expand1x1(x)
        out2: torch.Tensor = self.expand3x3(x)

        # Concatenate along channel dim
        x: torch.Tensor = torch.cat([out1, out2], dim=1)

        # Final activation
        x: torch.Tensor = self.expand_activation(x)
        return x
