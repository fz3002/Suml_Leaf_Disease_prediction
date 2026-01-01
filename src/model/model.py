import torch
import torch.nn as nn
from src.model.blocks import FireBlock


class SqueezeNet(nn.Module):
    """
    Fully convolutional neural network for image classification.

    This implementation follows SqueezeNet idea: replace many expensive
    3x3 convolutions with a composition of 1x1 convolutions and FireBlocks to
    reduce the number of parameters, while maintaining good accuracy.

    Architecture overview
    ---------------------
    - Feature extractor (`self.features`): Initial (Conv -> ReLU -> MaxPool) followed by a sequence of FireBlocks,
      with periodic MaxPool.
    - Classifier (`self.classifier`): Dropout -> 1x1 convolution mapping to num_classes -> global pooling
      via AdaptiveAvgPool2d to produce class logits.
    - The final 1x1 convolution (`self.final_conv`) produces a tensor of shape
      (N, num_classes, H, W), which is then reduced to (N, num_classes, 1, 1)
      using adaptive average pooling.
    - Weight initialization:
        - final_conv: Normal(mean=0, std=0.01)
        - other Conv2d: Kaiming/He initialization for ReLU
        - biases: zeros
    """

    def __init__(self, num_classes: int, dropout: float = 0.1, init_weights: bool = True) -> None:
        """
        SqueezeNet constructor

        :param num_classes: Number of output classes.
        :param dropout: Dropout probability used in the classifier section, by default 0.1.
        :param init_weights: If True, initialize convolution weights
        """

        super().__init__()
        # Feature extractor: stem + FireBlocks with occasional pooling
        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireBlock(in_channels=96, squeeze_channels=16, expand1x1_channels=64, expand3x3_channels=64),
            FireBlock(in_channels=128, squeeze_channels=16, expand1x1_channels=64, expand3x3_channels=64),
            FireBlock(in_channels=128, squeeze_channels=32, expand1x1_channels=128, expand3x3_channels=128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireBlock(in_channels=256, squeeze_channels=32, expand1x1_channels=128, expand3x3_channels=128),
            FireBlock(in_channels=256, squeeze_channels=48, expand1x1_channels=192, expand3x3_channels=192),
            FireBlock(in_channels=384, squeeze_channels=48, expand1x1_channels=192, expand3x3_channels=192),
            FireBlock(in_channels=384, squeeze_channels=64, expand1x1_channels=256, expand3x3_channels=256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireBlock(in_channels=512, squeeze_channels=64, expand1x1_channels=256, expand3x3_channels=256),
        )

        # Final 1x1 convolution that maps features to class channels
        self.final_conv: nn.Conv2d = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)

        # Classification head: regularize -> class projection -> global pooling
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Dropout(p=dropout),
            self.final_conv,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        if init_weights:
            self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        :param x: Input tensor
        :return: Output tensor
        """

        # 1. Extract Features
        x: torch.Tensor = self.features(x)

        # 2. Classify Features
        x: torch.Tensor = self.classifier(x)

        # (N, num_classes, 1, 1) -> (N, num_classes)
        x: torch.Tensor = torch.flatten(x, 1)
        return x

    def _init_weights(self) -> None:
        """
        Initialize the weights of the model. For last Conv layer, use Normal, for all other conv layers, use kaiming
        for ReLU. For biases initialize to 0.

        :return: None
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final_conv:
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def print_info(self) -> None:
        """
        Print a summary of parameter counts and estimated memory usage.

        :return: None
        """
        total_params = 0
        trainable_params = 0
        param_memory = 0
        grad_memory = 0

        for p in self.parameters():
            numel = p.numel()
            total_params += numel
            param_memory += numel * p.element_size()

            if p.requires_grad:
                trainable_params += numel
                if p.grad is not None:
                    grad_memory += numel * p.grad.element_size()

        buffer_memory = 0
        for b in self.buffers():
            buffer_memory += b.numel() * b.element_size()

        total_memory = param_memory + grad_memory + buffer_memory

        print("===== SqueezeNet Model Info =====")
        print(f"Total parameters:      {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("---------------------------------")
        print(f"Parameters memory: {param_memory / 1024 ** 2:.2f} MB")
        print(f"Gradients memory:  {grad_memory / 1024 ** 2:.2f} MB")
        print(f"Buffers memory:    {buffer_memory / 1024 ** 2:.2f} MB")
        print("---------------------------------")
        print(f"Total memory:      {total_memory / 1024 ** 2:.2f} MB")
        print("=================================")
