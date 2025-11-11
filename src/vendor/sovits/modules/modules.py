from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .dsconv import (
    DepthwiseSeparableConv1d,
    DepthwiseSeparableTransposeConv1d,
    remove_weight_norm_modules,
    weight_norm_modules,
)
from .. import attentions, commons

LRELU_SLOPE = 0.1

Conv1dModel = nn.Conv1d


def set_Conv1dModel(use_depthwise_conv: bool) -> None:
    global Conv1dModel
    Conv1dModel = DepthwiseSeparableConv1d if use_depthwise_conv else nn.Conv1d


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConvReluNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.in_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p_dropout)
        self.relu = nn.ReLU()

        self.in_layers.append(Conv1dModel(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))

        for _ in range(n_layers - 1):
            self.in_layers.append(Conv1dModel(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))

        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        for conv, norm in zip(self.in_layers, self.norm_layers):
            x = conv(x * x_mask)
            x = norm(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = residual + self.proj(x)
        return x * x_mask


class WN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1

        self.hidden_channels = hidden_channels
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.dropout = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = weight_norm_modules(cond_layer, name="weight")
        else:
            self.cond_layer = None

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = Conv1dModel(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding)
            in_layer = weight_norm_modules(in_layer, name="weight")
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = weight_norm_modules(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None and self.cond_layer is not None:
            g = self.cond_layer(g)
        elif self.cond_layer is not None:
            g = torch.zeros(1, 2 * self.hidden_channels * self.n_layers, x.size(-1), device=x.device, dtype=x.dtype)
        else:
            g = torch.zeros(1, 2 * self.hidden_channels * self.n_layers, x.size(-1), device=x.device, dtype=x.dtype)

        for i, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            x_in = in_layer(x)
            cond_offset = i * 2 * self.hidden_channels
            g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.dropout(acts)
            res_skip = res_skip_layer(acts)

            if i < self.n_layers - 1:
                res = res_skip[:, : self.hidden_channels, :]
                x = (x + res) * x_mask
                output = output + res_skip[:, self.hidden_channels :, :]
            else:
                output = output + res_skip

        return output * x_mask

    def remove_weight_norm(self) -> None:
        if self.cond_layer is not None:
            remove_weight_norm_modules(self.cond_layer)
        for layer in self.in_layers:
            remove_weight_norm_modules(layer)
        for layer in self.res_skip_layers:
            remove_weight_norm_modules(layer)


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int, int] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm_modules(
                    Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[0], padding=commons.get_padding(kernel_size, dilation[0]))
                ),
                weight_norm_modules(
                    Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[1], padding=commons.get_padding(kernel_size, dilation[1]))
                ),
                weight_norm_modules(
                    Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[2], padding=commons.get_padding(kernel_size, dilation[2]))
                ),
            ]
        )
        self.convs1.apply(commons.init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=1, padding=commons.get_padding(kernel_size, 1))),
                weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=1, padding=commons.get_padding(kernel_size, 1))),
                weight_norm_modules(Conv1dModel(channels, channels, kernel_size, 1, dilation=1, padding=commons.get_padding(kernel_size, 1))),
            ]
        )
        self.convs2.apply(commons.init_weights)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_weight_norm_modules(layer)
        for layer in self.convs2:
            remove_weight_norm_modules(layer)


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int] = (1, 3)) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm_modules(
                    Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[0], padding=commons.get_padding(kernel_size, dilation[0]))
                ),
                weight_norm_modules(
                    Conv1dModel(channels, channels, kernel_size, 1, dilation=dilation[1], padding=commons.get_padding(kernel_size, dilation[1]))
                ),
            ]
        )
        self.convs.apply(commons.init_weights)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for conv in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = conv(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs:
            remove_weight_norm_modules(layer)


class Log(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        x = torch.exp(x) * x_mask
        return x


class Flip(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        *args,
        reverse: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            return x, logdet
        return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        x = (x - self.m) * torch.exp(-self.logs) * x_mask
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        mean_only: bool = False,
        wn_sharing_parameter: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.mean_only = mean_only

        self.pre = nn.Conv1d(channels // 2, hidden_channels, 1)
        if wn_sharing_parameter is None:
            self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, p_dropout=0)
        else:
            self.enc = wn_sharing_parameter
        self.post = nn.Conv1d(hidden_channels, channels if mean_only else channels * 2, 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        x0, x1 = torch.split(x, self.channels // 2, dim=1)
        h = self.pre(x0)
        h = self.enc(h, x_mask, g=g)
        h = self.post(h)
        if self.mean_only:
            m = h
            logs = torch.zeros_like(m)
        else:
            m, logs = torch.split(h, self.channels // 2, dim=1)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], dim=1)
            logdet = torch.sum(logs * x_mask, [1, 2])
            return x, logdet
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        x = torch.cat([x0, x1], dim=1)
        return x


class TransformerCouplingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        n_layers: int,
        n_heads: int,
        p_dropout: float,
        filter_channels: int,
        mean_only: bool = False,
        wn_sharing_parameter: Optional[nn.Module] = None,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.mean_only = mean_only

        self.pre = nn.Conv1d(channels // 2, hidden_channels, 1)
        if wn_sharing_parameter is None:
            self.enc = attentions.FFT(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers=n_layers,
                kernel_size=kernel_size,
                p_dropout=p_dropout,
                isflow=True,
                gin_channels=gin_channels,
            )
        else:
            self.enc = wn_sharing_parameter
        self.post = nn.Conv1d(hidden_channels, channels if mean_only else channels * 2, 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        x0, x1 = torch.split(x, self.channels // 2, dim=1)
        h = self.pre(x0)
        h = self.enc(h, x_mask, g=g)
        h = self.post(h)

        if self.mean_only:
            m = h
            logs = torch.zeros_like(m)
        else:
            m, logs = torch.split(h, self.channels // 2, dim=1)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], dim=1)
            logdet = torch.sum(logs * x_mask, [1, 2])
            return x, logdet
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        x = torch.cat([x0, x1], dim=1)
        return x


class F0Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        spk_channels: int = 0,
    ) -> None:
        super().__init__()
        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = attentions.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1) if spk_channels != 0 else None

    def forward(
        self,
        x: torch.Tensor,
        norm_f0: torch.Tensor,
        x_mask: torch.Tensor,
        spk_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.detach(x)
        if spk_emb is not None and self.cond is not None:
            x = x + self.cond(spk_emb)
        x = x + self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x
