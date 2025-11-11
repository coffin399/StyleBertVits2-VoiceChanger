from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import torch
import torchaudio
from loguru import logger

from app.audio.capture import AudioCapture, CaptureSettings
from app.config import AppConfig
from app.output.opus_streamer import OpusSettings, OpusStreamer
from app.output.system_playback import PlaybackSettings, SystemPlayback
from app.vc.content_encoder import JapaneseHubertContentEncoder
from app.vc.f0 import F0Extractor
from app.vc.generator import GeneratorBackend


@dataclass(slots=True)
class EmotionState:
    label_scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ProsodyState:
    features: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class VoicePipelineState:
    last_transcript: str = ""
    last_emotion: Optional[EmotionState] = None
    last_prosody: Optional[ProsodyState] = None
    last_fusion_metadata: Optional[dict[str, float]] = None
    last_opus_packets: list[bytes] = field(default_factory=list)


class VoiceProcessingPipeline:
    """Real-time voice conversion pipeline orchestrating capture → VC → output."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        self._generator = GeneratorBackend(config.models, config.backend)
        self._generator_loaded = False
        self._content_encoder = JapaneseHubertContentEncoder(
            config.models, config.audio, device=config.backend.device
        )
        self._f0_extractor = F0Extractor(
            sample_rate=config.audio.input_sample_rate,
            frame_ms=config.streaming.frame_ms,
        )

        self._capture: Optional[AudioCapture] = None
        self._playback: Optional[SystemPlayback] = None
        self._opus_streamer: Optional[OpusStreamer] = None
        self._resampler: Optional[torchaudio.transforms.Resample] = None

        self._running = threading.Event()
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=32)
        self._worker: Optional[threading.Thread] = None

        self._frame_samples = self._calc_frame_samples()
        self._segment_samples = self._frame_samples * max(1, self._config.streaming.lookahead_frames)
        self._hop_samples = max(
            1,
            int(self._frame_samples * (1.0 - min(max(self._config.streaming.overlap_ratio, 0.0), 0.95))),
        )
        self._buffer = np.zeros(0, dtype=np.float32)

        self._state = VoicePipelineState()
        self._state_lock = threading.Lock()
        self._processed_chunks = 0
        self._dropped_blocks = 0
        self._target_speaker = 0

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running.is_set():
            return

        if not self._generator_loaded:
            logger.info("Loading generator backend…")
            self._generator.load()
            self._generator_loaded = True
            self._prepare_resampler()

        capture_settings = CaptureSettings(
            device_index=self._config.input_device_index,
            samplerate=float(self._config.audio.input_sample_rate),
            blocksize=self._frame_samples,
            channels=1,
            dtype="float32",
        )
        self._capture = AudioCapture(capture_settings)
        self._capture.register_callback(self._on_audio_block)

        playback_settings = PlaybackSettings(
            device_index=self._config.output_device_index,
            samplerate=self._config.audio.output_sample_rate,
            channels=1,
            blocksize=self._frame_samples,
        )
        self._playback = SystemPlayback(playback_settings)
        self._playback.start()

        self._opus_streamer = OpusStreamer(
            OpusSettings(
                sample_rate=self._config.audio.output_sample_rate,
                bitrate=self._config.audio.opus_bitrate,
            )
        )

        self._running.set()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._capture.start()
        logger.info("Voice pipeline started (frame={} samples, hop={} samples)", self._frame_samples, self._hop_samples)

    def stop(self) -> None:
        if not self._running.is_set():
            return

        self._running.clear()
        if self._capture is not None:
            self._capture.stop()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

        if self._playback is not None:
            self._playback.stop()
            self._playback = None

        if self._opus_streamer is not None:
            final_packet = self._opus_streamer.flush()
            if final_packet:
                with self._state_lock:
                    self._state.last_opus_packets.append(final_packet)

        self._buffer = np.zeros(0, dtype=np.float32)
        logger.info("Voice pipeline stopped. Processed chunks={} dropped_blocks={}", self._processed_chunks, self._dropped_blocks)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def state(self) -> VoicePipelineState:
        with self._state_lock:
            snapshot = replace(self._state)
            if snapshot.last_fusion_metadata is not None:
                snapshot.last_fusion_metadata = dict(snapshot.last_fusion_metadata)
            snapshot.last_opus_packets = list(snapshot.last_opus_packets)
            return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _calc_frame_samples(self) -> int:
        samples = int(self._config.audio.input_sample_rate * (self._config.streaming.frame_ms / 1000.0))
        return max(samples, 256)

    def _prepare_resampler(self) -> None:
        generator_sr = self._generator.sample_rate or self._config.audio.output_sample_rate
        if generator_sr != self._config.audio.output_sample_rate:
            self._resampler = torchaudio.transforms.Resample(
                orig_freq=generator_sr,
                new_freq=self._config.audio.output_sample_rate,
            )
            logger.info("Configured resampler {} Hz → {} Hz", generator_sr, self._config.audio.output_sample_rate)
        else:
            self._resampler = None

    def _on_audio_block(self, block: np.ndarray, _frames: int) -> None:
        mono = block.astype(np.float32, copy=False)
        if mono.ndim > 1:
            mono = np.mean(mono, axis=1)
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            self._dropped_blocks += 1

    def _worker_loop(self) -> None:
        while self._running.is_set() or not self._queue.empty():
            try:
                block = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if block is None:
                self._queue.task_done()
                break

            self._buffer = np.concatenate((self._buffer, block))
            while self._buffer.size >= self._segment_samples:
                chunk = self._buffer[: self._segment_samples].copy()
                self._buffer = self._buffer[self._hop_samples :]
                try:
                    self._process_chunk(chunk)
                except Exception as exc:  # pragma: no cover - runtime safeguard
                    logger.exception("Error during VC chunk processing: {}", exc)
                finally:
                    self._processed_chunks += 1
            self._queue.task_done()

    def _process_chunk(self, chunk: np.ndarray) -> None:
        if not self._generator_loaded or chunk.size == 0:
            return

        audio_tensor = torch.from_numpy(chunk).unsqueeze(0)
        content_features = self._content_encoder.encode(
            audio_tensor,
            sample_rate=self._config.audio.input_sample_rate,
        ).detach()

        f0 = self._f0_extractor.extract(chunk)

        waveform = self._generator.infer(content_features, f0, speaker_id=self._target_speaker)
        if waveform.size == 0:
            return

        output_wave = waveform
        if self._resampler is not None:
            resampled = self._resampler(torch.from_numpy(waveform).unsqueeze(0))
            output_wave = resampled.squeeze(0).cpu().numpy()
        output_wave = np.clip(output_wave, -1.0, 1.0)

        if self._playback is not None:
            self._playback.enqueue([np.expand_dims(output_wave, axis=1)])

        if self._opus_streamer is not None:
            packets = self._opus_streamer.encode(output_wave)
        else:
            packets = []

        metadata = {
            "chunks": float(self._processed_chunks + 1),
            "queue_depth": float(self._queue.qsize()),
            "dropped_blocks": float(self._dropped_blocks),
        }
        with self._state_lock:
            self._state.last_opus_packets = list(packets)
            self._state.last_fusion_metadata = metadata

*** End of File
