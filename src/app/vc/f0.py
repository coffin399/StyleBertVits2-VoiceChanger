from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyworld
import torch
from torch import Tensor


@dataclass(slots=True)
class F0ExtractorConfig:
    sample_rate: int
    frame_ms: float

    @property
    def frame_period(self) -> float:
        return self.frame_ms


class F0Extractor:
    """Streaming fundamental frequency extractor using WORLD Harvest."""

    def __init__(self, sample_rate: int, frame_ms: float) -> None:
        self._config = F0ExtractorConfig(sample_rate=sample_rate, frame_ms=frame_ms)

    def extract(self, audio: np.ndarray) -> Tensor:
        if audio.ndim > 1:
            audio = audio.squeeze(0)
        audio = audio.astype(np.float64, copy=False)
        f0, _ = pyworld.harvest(
            audio,
            self._config.sample_rate,
            frame_period=self._config.frame_period,
        )
        f0 = pyworld.stonemask(audio, f0, np.arange(len(f0)) * self._config.frame_period / 1000.0, self._config.sample_rate)
        return torch.from_numpy(f0.astype(np.float32))
