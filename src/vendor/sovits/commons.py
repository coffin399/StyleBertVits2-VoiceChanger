from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
from torch.nn import functional as F


def slice_pitch_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments_with_pitch(
    x: torch.Tensor,
    pitch: torch.Tensor,
    x_lengths: torch.Tensor | int | None = None,
    segment_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, _, t = x.size()
    if x_lengths is None:
        x_lengths = torch.full((b,), t, dtype=torch.long, device=x.device)
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    ret_pitch = slice_pitch_segments(pitch, ids_str, segment_size)
    return ret, ret_pitch, ids_str


def init_weights(module: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    classname = module.__class__.__name__
    if "Depthwise_Separable" in classname:
        module.depth_conv.weight.data.normal_(mean, std)
        module.point_conv.weight.data.normal_(mean, std)
    elif classname.find("Conv") != -1:  # type: ignore[attr-defined]
        module.weight.data.normal_(mean, std)  # type: ignore[attr-defined]


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape: Iterable[Tuple[int, int]]) -> list[int]:
    l = list(pad_shape)[::-1]
    return [item for sublist in l for item in sublist]


def intersperse(lst: Iterable[int], item: int) -> list[int]:
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = list(lst)
    return result


def kl_divergence(m_p: torch.Tensor, logs_p: torch.Tensor, m_q: torch.Tensor, logs_q: torch.Tensor) -> torch.Tensor:
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    return kl


def rand_gumbel(shape: torch.Size) -> torch.Tensor:
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x: torch.Tensor) -> torch.Tensor:
    return rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4) -> torch.Tensor:
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor | int | None = None,
    segment_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, _, t = x.size()
    if x_lengths is None:
        x_lengths = torch.full((b,), t, dtype=torch.long, device=x.device)
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def rand_spec_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor | int | None = None,
    segment_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, _, t = x.size()
    if x_lengths is None:
        x_lengths = torch.full((b,), t, dtype=torch.long, device=x.device)
    ids_str_max = x_lengths - segment_size
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length: int, channels: int, min_timescale: float = 1.0, max_timescale: float = 1.0e4) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1)
    inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(x: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4) -> torch.Tensor:
    _, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4, axis: int = 1) -> torch.Tensor:
    _, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length: int) -> torch.Tensor:
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor) -> torch.Tensor:
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def shift_1d(x: torch.Tensor) -> torch.Tensor:
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
