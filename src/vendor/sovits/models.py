from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from . import attentions, commons, modules, utils
from .commons import get_padding
from .utils import f0_to_coarse


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
        share_parameter: bool = False,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = (
            modules.WN(
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                p_dropout=0,
                gin_channels=gin_channels,
            )
            if share_parameter
            else None
        )

        for _ in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                )
            )
            self.flows.append(modules.Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class TransformerCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
        share_parameter: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = (
            attentions.FFT(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=self.gin_channels,
            )
            if share_parameter
            else None
        )

        for _ in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                    gin_channels=self.gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        n_layers: int,
        gin_channels: int = 0,
        filter_channels: Optional[int] = None,
        n_heads: Optional[int] = None,
        p_dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)

        self.enc_ = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        noice_scale: float = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if f0 is not None:
            x = x + self.f0_emb(f0).transpose(1, 2)
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale) * x_mask
        return z, m, logs, x_mask


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, use_spectral_norm: bool = False) -> None:
        super().__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv2d(
                        1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0)
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0)
                    )
                ),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrnMs256NSFsid(nn.Module):
    """so-vits-svc Synthesizer with NSF sid conditioning."""

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int,
        ssl_dim: int,
        n_speakers: int,
        sampling_rate: int,
        vol_embedding: bool = False,
        vocoder_name: str = "nsf-hifigan",
        use_depthwise_conv: bool = False,
        use_automatic_f0_prediction: bool = True,
        flow_share_parameter: bool = False,
        n_flow_layer: int = 4,
        n_layers_trans_flow: int = 3,
        use_transformer_flow: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.vol_embedding = vol_embedding
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.use_depthwise_conv = use_depthwise_conv
        self.use_automatic_f0_prediction = use_automatic_f0_prediction
        self.n_layers_trans_flow = n_layers_trans_flow
        if vol_embedding:
            self.emb_vol = nn.Linear(1, hidden_channels)

        self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

        self.emb_uv = nn.Embedding(2, hidden_channels)
        self.character_mix = False

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )

        hps = {
            "sampling_rate": sampling_rate,
            "inter_channels": inter_channels,
            "resblock": resblock,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels,
            "use_depthwise_conv": use_depthwise_conv,
        }

        modules.set_Conv1dModel(self.use_depthwise_conv)

        if vocoder_name == "nsf-hifigan":
            from .vdecoder.hifigan.models import Generator as Vocoder
        elif vocoder_name == "nsf-snake-hifigan":
            from .vdecoder.hifiganwithsnake.models import Generator as Vocoder
        else:
            from .vdecoder.hifigan.models import Generator as Vocoder

        self.dec = Vocoder(h=hps)

        self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)

        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers_trans_flow,
                5,
                p_dropout,
                n_flow_layer,
                gin_channels=gin_channels,
                share_parameter=flow_share_parameter,
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                n_flow_layer,
                gin_channels=gin_channels,
                share_parameter=flow_share_parameter,
            )

        if self.use_automatic_f0_prediction:
            self.f0_decoder = modules.F0Decoder(
                1,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                spk_channels=gin_channels,
            )
        else:
            self.f0_decoder = None

    def EnableCharacterMix(self, n_speakers_map: int, device: torch.device) -> None:
        speaker_map = torch.zeros((n_speakers_map, 1, 1, self.gin_channels)).to(device)
        for i in range(n_speakers_map):
            speaker_map[i] = self.emb_g(torch.LongTensor([[i]]).to(device))
        self.speaker_map = speaker_map.unsqueeze(0).to(device)
        self.character_mix = True

    def forward(
        self,
        c: torch.Tensor,
        f0: torch.Tensor,
        uv: torch.Tensor,
        spec: torch.Tensor,
        g: torch.Tensor,
        c_lengths: torch.Tensor,
        spec_lengths: torch.Tensor,
        vol: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.emb_g(g).transpose(1, 2)
        vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) if vol is not None and self.vol_embedding else 0

        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        if self.use_automatic_f0_prediction and self.f0_decoder is not None:
            lf0 = 2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500.0
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
        else:
            pred_lf0 = norm_lf0 = lf0 = torch.zeros_like(f0)

        z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        z_p = self.flow(z, spec_mask, g=g)
        z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(z, f0, spec_lengths, self.segment_size)
        o = self.dec(z_slice, g=g, f0=pitch_slice)
        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0

    @torch.no_grad()
    def infer(
        self,
        c: torch.Tensor,
        f0: torch.Tensor,
        uv: torch.Tensor,
        g: torch.Tensor,
        noice_scale: float = 0.35,
        seed: int = 52468,
        predict_f0: bool = False,
        vol: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if c.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        if self.character_mix and len(g) > 1:
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))
            g = g * self.speaker_map
            g = torch.sum(g, dim=1)
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0)
        else:
            if g.dim() == 1:
                g = g.unsqueeze(0)
            g = self.emb_g(g).transpose(1, 2)

        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) if vol is not None and self.vol_embedding else 0
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol

        if self.use_automatic_f0_prediction and predict_f0 and self.f0_decoder is not None:
            lf0 = 2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500.0
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv, random_scale=False)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)

        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), noice_scale=noice_scale)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=f0)
        return o, f0
