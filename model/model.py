import torch
import torch.nn as nn
from model.blocks import FireBlock


class SqueezeNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 dropout: float = 0.1,
                 init_weights: bool = True,
                 ) -> None:
        super().__init__()
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

        self.final_conv: nn.Conv2d = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Dropout(p=dropout),
            self.final_conv,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        if init_weights:
            self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.features(x)
        x: torch.Tensor = self.classifier(x)
        x: torch.Tensor = torch.flatten(x, 1)
        return x

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final_conv:
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def print_info(self) -> None:
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
