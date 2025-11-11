from __future__ import annotations

import torch

F0_BIN = 256
F0_MAX = 1100.0
F0_MIN = 50.0
F0_MEL_MIN = 1127 * torch.log1p(torch.tensor(F0_MIN / 700.0))
F0_MEL_MAX = 1127 * torch.log1p(torch.tensor(F0_MAX / 700.0))


def normalize_f0(f0: torch.Tensor, x_mask: torch.Tensor, uv: torch.Tensor, random_scale: bool = True) -> torch.Tensor:
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    uv_sum = torch.where(uv_sum == 0, torch.full_like(uv_sum, 9999.0), uv_sum)
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

    if random_scale:
        factor = torch.empty(f0.shape[0], 1, device=f0.device, dtype=f0.dtype).uniform_(0.8, 1.2)
    else:
        factor = torch.ones(f0.shape[0], 1, device=f0.device, dtype=f0.dtype)

    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    f0_norm = torch.nan_to_num(f0_norm)
    return f0_norm * x_mask


def f0_to_coarse(f0: torch.Tensor) -> torch.Tensor:
    # guard zero values before log
    f0_mel = 1127.0 * torch.log1p(f0 / 700.0)
    a = (F0_BIN - 2) / (F0_MEL_MAX - F0_MEL_MIN)
    b = F0_MEL_MIN * a - 1.0
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = torch.clamp(f0_coarse, min=1, max=F0_BIN - 1)
    return f0_coarse
