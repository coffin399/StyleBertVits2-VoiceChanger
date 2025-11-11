from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from . import commons
from .modules.dsconv import weight_norm_modules
from .modules.modules import LayerNorm


class FFT(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int = 1,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        proximal_bias: bool = False,
        proximal_init: bool = True,
        isflow: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        if isflow:
            gin_channels = kwargs.get("gin_channels", 0)
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_pre = nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            self.cond_layer = weight_norm_modules(cond_layer, name="weight")
            self.gin_channels = gin_channels
        else:
            self.cond_layer = None
            self.cond_pre = None
            self.gin_channels = 0

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        for _ in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if g is not None and self.cond_layer is not None:
            g = self.cond_layer(g)
        else:
            g = None

        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
        x = x * x_mask
        for i in range(self.n_layers):
            if g is not None and self.cond_pre is not None:
                x_cond = self.cond_pre(x)
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
                x = commons.fused_add_tanh_sigmoid_multiply(
                    x_cond,
                    g_l,
                    torch.IntTensor([self.hidden_channels]).to(x.device),
                )
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
        return x * x_mask


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        proximal_bias: bool = False,
        proximal_init: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        h: torch.Tensor,
        h_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        heads_share: bool = True,
        block_length: Optional[int] = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ) -> None:
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn: Optional[torch.Tensor] = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
        else:
            self.emb_rel_k = None
            self.emb_rel_v = None

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        nn.init.xavier_uniform_(self.conv_o.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, d, t_s = key.size()
        t_t = query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention requires self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias requires self-attention."
            scores = scores + self._attention_bias_proximal(t_s)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        output = torch.matmul(attn, value)
        if self.window_size is not None and self.emb_rel_v is not None:
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(attn, value_relative_embeddings)

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, attn

    def _get_relative_embeddings(self, relative_embeddings: torch.Tensor, length: int) -> torch.Tensor:
        max_relative_position = 2 * self.window_size + 1
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max(self.window_size + 1 - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])
        else:
            padded_relative_embeddings = relative_embeddings
        return padded_relative_embeddings[:, slice_start_position:slice_end_position]

    def _matmul_with_relative_keys(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = x.view(batch * heads, length, -1)
        y = y.repeat(batch if y.size(0) == 1 else 1, 1, 1)
        y = y.view(batch * heads, -1, length)
        result = torch.bmm(x, y)
        result = result.view(batch, heads, length, length)
        return result

    def _relative_position_to_absolute_position(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, 1])
        x_flat = x.view(batch, heads, -1)
        x_flat = F.pad(x_flat, [length, 0])
        x = x_flat.view(batch, heads, length + 1, 2 * length)
        x = x[:, :, :, 1:].view(batch, heads, length, length)
        return x

    def _matmul_with_relative_values(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = x.view(batch * heads, length, -1)
        y = y.repeat(batch if y.size(0) == 1 else 1, 1, 1)
        y = y.view(batch * heads, -1, length)
        result = torch.bmm(x, y)
        result = result.view(batch, heads, length, -1)
        return result

    def _attention_bias_proximal(self, length: int) -> torch.Tensor:
        r = torch.arange(length, device=self.conv_q.weight.device, dtype=self.conv_q.weight.dtype)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return -torch.abs(diff)


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal

        padding = kernel_size - 1 if causal else (kernel_size - 1) // 2
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=padding)
        self.conv_2 = nn.Conv1d(filter_channels, hidden_channels, kernel_size, padding=padding)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x = F.pad(x, (self.conv_1.kernel_size[0] - 1, 0))
        x = self.conv_1(x)
        x = F.gelu(x)
        if self.causal:
            x = F.pad(x, (self.conv_2.kernel_size[0] - 1, 0))
        x = self.conv_2(x)
        x = self.drop(x)
        return x * x_mask
