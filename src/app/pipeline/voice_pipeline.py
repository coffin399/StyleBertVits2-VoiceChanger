from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional, Tuple

import numpy as np

from app.audio.capture import AudioCapture, AudioCallback, CaptureSettings
from app.audio.devices import AudioDevice
from app.config import AppConfig
from app.emotion.prosody_analyzer import ProsodyAnalyzer, ProsodyResult
from app.emotion.text_analyzer import TextEmotionAnalyzer, TextEmotionResult
from app.output.system_playback import PlaybackSettings, SystemPlayback
from app.style.fusion import EmotionFusion
from app.style.tts_engine import StyleBertSynthesizer, StyleModelPaths
from app.transcription.whisper_service import TranscriptionResult, WhisperService


@dataclass(slots=True)
class PipelineState:
    """Runtime state exposed to the GUI."""

    last_transcript: str = ""
    last_emotion: Optional[TextEmotionResult] = None
    last_prosody: Optional[ProsodyResult] = None
    last_fusion_metadata: Optional[dict[str, float]] = None


class VoiceProcessingPipeline:
    """Coordinate capture → transcription → emotion fusion → TTS → playback."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._state = PipelineState()

        # Audio capture setup
        capture_settings = CaptureSettings(
            device_index=config.input_device_index,
            samplerate=float(config.whisper.sample_rate),
            channels=1,
            dtype="float32",
        )
        self._capture = AudioCapture(capture_settings)

        # Transcription and emotion modules
        self._whisper = WhisperService(config.whisper)
        self._text_emotion = TextEmotionAnalyzer(config.emotion)
        self._prosody = ProsodyAnalyzer(
            config.emotion.opensmile_config,
            config.emotion.opensmile_feature_level,
        )

        # StyleBert synthesizer and fusion
        style_paths = StyleModelPaths(
            model_file=config.stylebert_model_file,
            config_file=config.stylebert_config_file,
            style_vectors=config.stylebert_style_vectors,
        )
        self._synthesizer = StyleBertSynthesizer(style_paths, config.style)
        base_style_vector = self._synthesizer.get_style_vector()
        self._fusion = EmotionFusion(config.emotion, base_style_vector)

        # Playback routing
        playback_settings = PlaybackSettings(
            device_index=config.output_device_index,
            samplerate=config.style.sample_rate,
            channels=1,
        )
        self._playback = SystemPlayback(playback_settings)

        # Async processing primitives
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop, name="voice-pipeline-loop", daemon=True
        )
        self._audio_queue: asyncio.Queue[Tuple[np.ndarray, float]] = asyncio.Queue()
        self._running = threading.Event()
        self._processing_task: Optional[asyncio.Future] = None

    @property
    def state(self) -> PipelineState:
        return self._state

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._loop_thread.start()
        self._processing_task = asyncio.run_coroutine_threadsafe(
            self._processing_loop(), self._loop
        )
        self._capture.register_callback(self._on_audio_block)
        self._capture.start()
        self._playback.start()

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        self._capture.stop()
        self._playback.stop()
        asyncio.run_coroutine_threadsafe(self._audio_queue.put((np.zeros(0), -1.0)), self._loop)
        if self._processing_task is not None:
            try:
                self._processing_task.result(timeout=5)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=2)

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        self._loop.close()

    def _on_audio_block(self, block: np.ndarray, frames: int) -> None:
        if not self._running.is_set():
            return
        mono = block[:, 0] if block.ndim > 1 else block
        mono = mono.astype(np.float32, copy=True)
        duration = frames / float(self._config.whisper.sample_rate)
        asyncio.run_coroutine_threadsafe(
            self._audio_queue.put((mono, duration)), self._loop
        )

    async def _processing_loop(self) -> None:
        buffer: Deque[np.ndarray] = deque()
        buffer_duration = 0.0
        sample_rate = self._config.whisper.sample_rate

        while self._running.is_set():
            audio_block, duration = await self._audio_queue.get()
            if duration < 0:
                self._audio_queue.task_done()
                break
            buffer.append(audio_block)
            buffer_duration += duration
            await self._whisper.add_audio_block(audio_block, duration)
            self._audio_queue.task_done()

            if buffer_duration < self._config.whisper.chunk_seconds:
                continue

            prosody_samples = (
                np.concatenate(list(buffer)) if buffer else np.zeros(0, dtype=np.float32)
            )
            async for transcript in self._whisper.transcribe():
                if not transcript.text:
                    continue
                await self._handle_transcript(transcript, prosody_samples, sample_rate)

            buffer.clear()
            buffer_duration = 0.0

    async def _handle_transcript(
        self,
        transcript: TranscriptionResult,
        prosody_samples: np.ndarray,
        sample_rate: int,
    ) -> None:
        text_emotion = self._text_emotion.analyze(transcript.text)
        prosody_result = self._prosody.analyze(prosody_samples, sample_rate)
        fusion_result = self._fusion.fuse(text_emotion, prosody_result)

        sr, audio = self._synthesizer.synthesize(
            text=transcript.text,
            style_vector=fusion_result.style_vector,
        )
        float_audio = audio.astype(np.float32) / 32768.0
        self._playback.enqueue([float_audio])

        self._state.last_transcript = transcript.text
        self._state.last_emotion = text_emotion
        self._state.last_prosody = prosody_result
        self._state.last_fusion_metadata = fusion_result.metadata


def get_default_devices() -> Tuple[Optional[AudioDevice], Optional[AudioDevice]]:
    """Convenience helper to fetch default input/output devices."""

    import sounddevice as sd  # Local import to avoid mandatory dependency at import time

    default_input = sd.default.device[0]
    default_output = sd.default.device[1]
    input_device = None
    output_device = None
    for idx, info in enumerate(sd.query_devices()):
        if idx == default_input:
            input_device = AudioDevice(
                name=info["name"],
                index=idx,
                max_input_channels=info["max_input_channels"],
                max_output_channels=info["max_output_channels"],
                default_samplerate=info["default_samplerate"],
            )
        if idx == default_output:
            output_device = AudioDevice(
                name=info["name"],
                index=idx,
                max_input_channels=info["max_input_channels"],
                max_output_channels=info["max_output_channels"],
                default_samplerate=info["default_samplerate"],
            )
    return input_device, output_device
