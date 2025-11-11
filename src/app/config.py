"""Configuration definitions for the Voice Conversion application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class StreamingConfig:
    """Parameters controlling real-time buffering."""

    frame_ms: int = 32
    lookahead_frames: int = 6
    overlap_ratio: float = 0.5


@dataclass(slots=True)
class BackendConfig:
    """Inference backend settings (GPU only)."""

    device: str = "cuda"
    cuda_visible_devices: Optional[str] = None
    prefer_tensorrt: bool = True
    onnx_provider: str = "TensorrtExecutionProvider"
    use_fp16: bool = True
    enable_int8: bool = False
    enable_kv_cache: bool = True


@dataclass(slots=True)
class ModelConfig:
    """Paths to VC generator and auxiliary models."""

    generator_checkpoint: Path = Path("models") / "generator.pth"
    generator_onnx: Optional[Path] = Path("models") / "generator.onnx"
    tensorrt_engine: Optional[Path] = Path("models") / "generator.engine"
    generator_config: Path = Path("models") / "config.json"
    content_encoder: Path = Path("models") / "content_encoder" / "model.safetensors"
    cluster_model: Optional[Path] = None
    speaker_map: Optional[Path] = None


@dataclass(slots=True)
class AudioConfig:
    input_sample_rate: int = 48000
    content_sample_rate: int = 16000
    output_sample_rate: int = 48000
    opus_bitrate: int = 64000


@dataclass(slots=True)
class AppConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model_root: Path = Path("models")
    cache_dir: Path = Path("cache")
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""

        for path in [
            self.model_root,
            self.models.content_encoder.parent,
            self.cache_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
