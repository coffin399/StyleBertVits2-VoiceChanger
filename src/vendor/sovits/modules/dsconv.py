from __future__ import annotations

import torch.nn as nn
from torch.nn.utils import remove_weight_norm, weight_norm


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.depth_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.point_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))

    def weight_norm(self) -> None:
        self.depth_conv = weight_norm(self.depth_conv, name="weight")
        self.point_conv = weight_norm(self.point_conv, name="weight")

    def remove_weight_norm(self) -> None:
        self.depth_conv = remove_weight_norm(self.depth_conv, name="weight")
        self.point_conv = remove_weight_norm(self.point_conv, name="weight")


class DepthwiseSeparableTransposeConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.depth_conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            output_padding=output_padding,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.point_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))

    def weight_norm(self) -> None:
        self.depth_conv = weight_norm(self.depth_conv, name="weight")
        self.point_conv = weight_norm(self.point_conv, name="weight")

    def remove_weight_norm(self) -> None:
        remove_weight_norm(self.depth_conv, name="weight")
        remove_weight_norm(self.point_conv, name="weight")


def weight_norm_modules(module: nn.Module, name: str = "weight", dim: int = 0) -> nn.Module:
    if isinstance(module, (DepthwiseSeparableConv1d, DepthwiseSeparableTransposeConv1d)):
        module.weight_norm()
        return module
    return weight_norm(module, name, dim)


def remove_weight_norm_modules(module: nn.Module, name: str = "weight") -> None:
    if isinstance(module, (DepthwiseSeparableConv1d, DepthwiseSeparableTransposeConv1d)):
        module.remove_weight_norm()
    else:
        remove_weight_norm(module, name)
