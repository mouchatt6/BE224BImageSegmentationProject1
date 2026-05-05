from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPlusPlus(nn.Module):
    """Four-level U-Net++ with dense decoder skip pathways."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        c = base_channels
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)

        self.conv0_0 = ConvBlock(in_channels, c)
        self.conv1_0 = ConvBlock(c, c * 2)
        self.conv2_0 = ConvBlock(c * 2, c * 4)
        self.conv3_0 = ConvBlock(c * 4, c * 8)
        self.conv4_0 = ConvBlock(c * 8, c * 16)

        self.conv0_1 = ConvBlock(c + c * 2, c)
        self.conv1_1 = ConvBlock(c * 2 + c * 4, c * 2)
        self.conv2_1 = ConvBlock(c * 4 + c * 8, c * 4)
        self.conv3_1 = ConvBlock(c * 8 + c * 16, c * 8)

        self.conv0_2 = ConvBlock(c * 2 + c * 2, c)
        self.conv1_2 = ConvBlock(c * 4 + c * 4, c * 2)
        self.conv2_2 = ConvBlock(c * 8 + c * 8, c * 4)

        self.conv0_3 = ConvBlock(c * 3 + c * 2, c)
        self.conv1_3 = ConvBlock(c * 6 + c * 4, c * 2)

        self.conv0_4 = ConvBlock(c * 4 + c * 2, c)

        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)
        if deep_supervision:
            self.out_conv1 = nn.Conv2d(c, out_channels, kernel_size=1)
            self.out_conv2 = nn.Conv2d(c, out_channels, kernel_size=1)
            self.out_conv3 = nn.Conv2d(c, out_channels, kernel_size=1)
            self.out_conv4 = nn.Conv2d(c, out_channels, kernel_size=1)

    @staticmethod
    def _up_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self._up_to(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up_to(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up_to(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up_to(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up_to(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up_to(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up_to(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up_to(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up_to(x2_2, x1_0)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up_to(x1_3, x0_0)], dim=1))

        if self.deep_supervision:
            return torch.stack(
                [
                    self.out_conv1(x0_1),
                    self.out_conv2(x0_2),
                    self.out_conv3(x0_3),
                    self.out_conv4(x0_4),
                ],
                dim=0,
            ).mean(dim=0)

        return self.out_conv(x0_4)


def build_unetpp(base_channels: int = 32, deep_supervision: bool = False) -> UNetPlusPlus:
    return UNetPlusPlus(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
    )

