"""Application configuration models and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class WhisperConfig:
    model_size: str = "large-v2"
    device: str = "auto"  # "cpu", "cuda", "auto"
    compute_type: str = "float16"
    beam_size: int = 5
    language: Optional[str] = "ja"
    vad_filter: bool = True
    vad_threshold: float = 0.6
    chunk_seconds: float = 5.0
    sample_rate: int = 16000


@dataclass(slots=True)
class EmotionConfig:
    text_model_name: str = "daigo/bert-base-japanese-sentiment"
    device: str = "auto"
    opensmile_config: str = "GeMAPSv01a"
    opensmile_feature_level: str = "Functionals"
    text_weight: float = 0.6
    prosody_weight: float = 0.4


@dataclass(slots=True)
class StyleConfig:
    speaker_id: int = 0
    style_name: str = "Neutral"
    style_weight: float = 1.0
    noise_scale: float = 0.6
    noise_w: float = 0.8
    length_scale: float = 1.0
    sdp_ratio: float = 0.2
    sample_rate: int = 22050
    reference_audio: Optional[Path] = None


@dataclass(slots=True)
class AppConfig:
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    model_root: Path = Path("models")
    stylebert_model_dir: Path = Path("models") / "default"
    stylebert_model_file: Path = Path("models") / "default" / "model.safetensors"
    stylebert_config_file: Path = Path("models") / "default" / "config.json"
    stylebert_style_vectors: Path = Path("models") / "default" / "style_vectors.npy"
    cache_dir: Path = Path("cache")
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None

    def ensure_directories(self) -> None:
        """Ensure that directories expected by the application exist."""

        for path in [
            self.model_root,
            self.stylebert_model_dir,
            self.cache_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
