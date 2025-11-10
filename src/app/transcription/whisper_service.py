from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np
from faster_whisper import WhisperModel

from app.config import WhisperConfig


def _auto_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    language: Optional[str]
    start: float
    end: float
    probability: float


class WhisperService:
    """Streaming transcription helper around faster-whisper."""

    def __init__(self, config: WhisperConfig) -> None:
        self._config = config
        device = _auto_device(config.device)
        compute_type = config.compute_type
        self._model = WhisperModel(
            config.model_size,
            device=device,
            compute_type=compute_type,
        )
        self._buffer = deque[tuple[np.ndarray, float]]()
        self._buffer_duration = 0.0
        self._lock = asyncio.Lock()

    async def add_audio_block(self, block: np.ndarray, duration: float) -> None:
        async with self._lock:
            self._buffer.append((block.copy(), duration))
            self._buffer_duration += duration
            while self._buffer_duration > self._config.chunk_seconds:
                old_block, old_dur = self._buffer.popleft()
                self._buffer_duration -= old_dur
                del old_block

    async def transcribe(self) -> AsyncIterator[TranscriptionResult]:
        """Yield transcription results for the current buffer."""

        async with self._lock:
            if not self._buffer:
                return
            samples = np.concatenate([b for b, _ in self._buffer], axis=0)
        segments, info = self._model.transcribe(
            samples,
            beam_size=self._config.beam_size,
            language=self._config.language,
            vad_filter=self._config.vad_filter,
            vad_parameters={"threshold": self._config.vad_threshold},
        )
        for segment in segments:
            yield TranscriptionResult(
                text=segment.text.strip(),
                language=segment.language or info.language,
                start=segment.start,
                end=segment.end,
                probability=segment.avg_logprob,
            )
