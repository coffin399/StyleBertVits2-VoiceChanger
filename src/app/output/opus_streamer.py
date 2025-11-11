from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from opuslib import Encoder, APPLICATION_AUDIO


@dataclass(slots=True)
class OpusSettings:
    sample_rate: int
    channels: int = 1
    bitrate: int = 64000


class OpusStreamer:
    """Encode PCM frames into Opus packets."""

    def __init__(self, settings: OpusSettings) -> None:
        self._settings = settings
        self._encoder = Encoder(settings.sample_rate, settings.channels, APPLICATION_AUDIO)
        self._encoder.bitrate = settings.bitrate
        self._frame_samples = int(settings.sample_rate * 0.02)  # 20 ms default frame size
        self._buffer: list[float] = []

    def encode(self, samples: np.ndarray) -> list[bytes]:
        if samples.ndim > 1:
            samples = samples[:, 0]
        pcm = samples.astype(np.float32).tolist()
        self._buffer.extend(pcm)
        packets: list[bytes] = []
        while len(self._buffer) >= self._frame_samples:
            frame = np.array(self._buffer[: self._frame_samples], dtype=np.float32)
            del self._buffer[: self._frame_samples]
            packets.append(self._encoder.encode_float(frame.tobytes(), self._frame_samples))
        return packets

    def flush(self) -> Optional[bytes]:
        if not self._buffer:
            return None
        frame = np.array(self._buffer, dtype=np.float32)
        self._buffer.clear()
        return self._encoder.encode_float(frame.tobytes(), len(frame))
