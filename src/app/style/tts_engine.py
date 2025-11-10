from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from app.config import StyleConfig
from vendor.style_bert_vits2.style_bert_vits2.tts_model import TTSModel
from vendor.style_bert_vits2.style_bert_vits2.constants import Languages


@dataclass(slots=True)
class StyleModelPaths:
    model_file: Path
    config_file: Path
    style_vectors: Path


class StyleBertSynthesizer:
    """Wrapper around the vendorized StyleBertVits2 TTSModel."""

    def __init__(self, paths: StyleModelPaths, style_config: StyleConfig) -> None:
        self._paths = paths
        self._style_config = style_config
        self._model: Optional[TTSModel] = None

    def _ensure_model(self) -> TTSModel:
        if self._model is None:
            self._model = TTSModel(
                model_path=self._paths.model_file,
                config_path=self._paths.config_file,
                style_vec_path=self._paths.style_vectors,
                device="cuda" if self._style_config.speaker_id >= 0 else "cpu",
            )
        return self._model

    def get_style_vector(self, style_name: Optional[str] = None, weight: Optional[float] = None) -> np.ndarray:
        model = self._ensure_model()
        style_name = style_name or self._style_config.style_name
        weight = weight if weight is not None else self._style_config.style_weight
        style_id = model.style2id.get(style_name, 0)
        vector = model.get_style_vector(style_id, weight)
        return vector.astype(np.float32, copy=True)

    def synthesize(
        self,
        text: str,
        style_vector: Optional[np.ndarray] = None,
        speaker_id: Optional[int] = None,
        sdp_ratio: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        length_scale: Optional[float] = None,
    ) -> Tuple[int, np.ndarray]:
        model = self._ensure_model()
        style_vector = style_vector if style_vector is not None else self.get_style_vector()
        speaker_id = speaker_id if speaker_id is not None else self._style_config.speaker_id
        sdp_ratio = sdp_ratio if sdp_ratio is not None else self._style_config.sdp_ratio
        noise_scale = noise_scale if noise_scale is not None else self._style_config.noise_scale
        noise_w = noise_w if noise_w is not None else self._style_config.noise_w
        length_scale = length_scale if length_scale is not None else self._style_config.length_scale

        sampling_rate, audio = model.infer(
            text=text,
            language=Languages.JP,
            speaker_id=speaker_id,
            sdp_ratio=sdp_ratio,
            noise=noise_scale,
            noise_w=noise_w,
            length=length_scale,
            style_weight=self._style_config.style_weight,
            style=self._style_config.style_name,
            given_phone=None,
            given_tone=None,
            style_vector_override=style_vector,
        )
        return sampling_rate, audio
