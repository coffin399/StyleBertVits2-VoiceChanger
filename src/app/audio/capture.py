from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


AudioCallback = Callable[[np.ndarray, int], None]


@dataclass(slots=True)
class CaptureSettings:
    device_index: Optional[int] = None
    samplerate: float = 16000.0
    blocksize: int = 0
    channels: int = 1
    dtype: str = "float32"


class AudioCapture:
    """Manage microphone capture and stream audio blocks to registered callbacks."""

    def __init__(self, settings: CaptureSettings) -> None:
        self._settings = settings
        self._callbacks: list[AudioCallback] = []
        self._stream: Optional[sd.InputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self._running = threading.Event()

    def register_callback(self, callback: AudioCallback) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        if self._stream is not None:
            return
        self._running.set()
        self._stream = sd.InputStream(
            device=self._settings.device_index,
            samplerate=self._settings.samplerate,
            channels=self._settings.channels,
            blocksize=self._settings.blocksize,
            dtype=self._settings.dtype,
            callback=self._on_audio,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _on_audio(self, in_data: np.ndarray, frames: int, *_args) -> None:
        if not self._running.is_set():
            return
        try:
            self._queue.put_nowait(np.copy(in_data))
        except queue.Full:
            # Drop if processing can't keep up
            pass

    def _dispatch_loop(self) -> None:
        while self._running.is_set():
            try:
                block = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            for callback in self._callbacks:
                try:
                    callback(block, len(block))
                except Exception:  # pragma: no cover - log upstream
                    continue
