from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import sounddevice as sd


@dataclass(slots=True)
class PlaybackSettings:
    device_index: Optional[int] = None
    samplerate: int = 22050
    channels: int = 1
    blocksize: int = 1024
    dtype: str = "float32"


class SystemPlayback:
    """Simple buffered audio playback using sounddevice."""

    def __init__(self, settings: PlaybackSettings) -> None:
        self._settings = settings
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
        self._stream: Optional[sd.OutputStream] = None

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            device=self._settings.device_index,
            samplerate=self._settings.samplerate,
            channels=self._settings.channels,
            blocksize=self._settings.blocksize,
            dtype=self._settings.dtype,
            callback=self._on_consume,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def enqueue(self, samples: Iterable[np.ndarray]) -> None:
        for block in samples:
            block = np.asarray(block, dtype=np.float32)
            if block.ndim == 1:
                block = np.expand_dims(block, axis=1)
            try:
                self._queue.put_nowait(block)
            except queue.Full:
                break

    def _on_consume(self, out_data: np.ndarray, frames: int, *_args) -> None:
        try:
            chunk = self._queue.get_nowait()
        except queue.Empty:
            out_data.fill(0)
            return
        if chunk.shape[0] < frames:
            padded = np.zeros((frames, chunk.shape[1]), dtype=chunk.dtype)
            padded[: chunk.shape[0], :] = chunk
            chunk = padded
        out_data[:] = chunk[:frames]
        self._queue.task_done()
