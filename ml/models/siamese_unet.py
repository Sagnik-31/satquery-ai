"""Compact shared-encoder Siamese U-Net for binary change segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class SharedEncoder(nn.Module):
    """Four downsampling stages, giving a total spatial factor of 16."""

    def __init__(self, input_channels: int, base_channels: int):
        super().__init__()
        channels = [base_channels * (2**index) for index in range(5)]
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList(
            [
                ResidualBlock(channels[0], channels[0]),
                ResidualBlock(channels[0], channels[1], stride=2),
                ResidualBlock(channels[1], channels[2], stride=2),
                ResidualBlock(channels[2], channels[3], stride=2),
                ResidualBlock(channels[3], channels[4], stride=2),
            ]
        )
        self.channels = channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class FeatureFusion(nn.Module):
    """Fuse same-scale T1/T2 features and their absolute temporal difference."""

    def __init__(self, channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResidualBlock(channels, channels),
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat((t1, t2, torch.abs(t1 - t2)), dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class SiameseUNet(nn.Module):
    """Shared-encoder, multi-scale-fusion change model with one output logit.

    Inputs are separate tensors so temporal weight sharing is explicit:
    ``model(t1, t2)``. The model intentionally has no pretrained-weight path.
    """

    downsampling_factor = 16

    def __init__(self, input_channels: int = 3, base_channels: int = 24):
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        self.input_channels = input_channels
        self.encoder = SharedEncoder(input_channels, base_channels)
        self.fusions = nn.ModuleList(FeatureFusion(channels) for channels in self.encoder.channels)
        channels = self.encoder.channels
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(channels[4], channels[3], channels[3]),
                DecoderBlock(channels[3], channels[2], channels[2]),
                DecoderBlock(channels[2], channels[1], channels[1]),
                DecoderBlock(channels[1], channels[0], channels[0]),
            ]
        )
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def _validate_inputs(self, t1: torch.Tensor, t2: torch.Tensor) -> None:
        if t1.ndim != 4 or t2.ndim != 4:
            raise ValueError("SiameseUNet expects T1 and T2 tensors shaped [B, C, H, W].")
        if t1.shape != t2.shape:
            raise ValueError(f"T1 and T2 must have identical shapes; got {tuple(t1.shape)} and {tuple(t2.shape)}.")
        if t1.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {t1.shape[1]}.")
        height, width = t1.shape[-2:]
        if height % self.downsampling_factor or width % self.downsampling_factor:
            raise ValueError(
                f"Input spatial dimensions must be divisible by {self.downsampling_factor}; got {height}x{width}."
            )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(t1, t2)
        t1_features = self.encoder(t1)
        t2_features = self.encoder(t2)
        fused = [fusion(left, right) for fusion, left, right in zip(self.fusions, t1_features, t2_features)]
        x = fused[-1]
        for decoder, skip in zip(self.decoders, reversed(fused[:-1])):
            x = decoder(x, skip)
        return self.head(x)
