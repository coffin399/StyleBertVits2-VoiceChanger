from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from app.config import EmotionConfig
from app.emotion.prosody_analyzer import ProsodyResult
from app.emotion.text_analyzer import TextEmotionResult


@dataclass(slots=True)
class EmotionFusionResult:
    style_vector: np.ndarray
    metadata: Dict[str, float]


class EmotionFusion:
    """Combine text emotion scores and prosodic features into a style conditioning vector."""

    def __init__(self, config: EmotionConfig, base_style_vector: np.ndarray) -> None:
        self._config = config
        self._base_style_vector = base_style_vector.astype(np.float32)
        self._dimension = int(self._base_style_vector.size)

    def fuse(
        self,
        text_emotion: TextEmotionResult,
        prosody: ProsodyResult,
    ) -> EmotionFusionResult:
        text_weight = max(0.0, min(1.0, self._config.text_weight))
        prosody_weight = max(0.0, min(1.0 - text_weight, self._config.prosody_weight))
        base_weight = max(0.0, 1.0 - text_weight - prosody_weight)

        norm_text_vector = _normalize_scores(text_emotion.label_scores, self._dimension)
        prosody_vector = _prosody_to_vector(prosody.features, self._dimension)

        fused = (
            self._base_style_vector * base_weight
            + norm_text_vector * text_weight
            + prosody_vector * prosody_weight
        )
        metadata: Dict[str, float] = {f"text_{k}": v for k, v in text_emotion.label_scores.items()}
        metadata.update({f"prosody_{k}": v for k, v in prosody.features.items()})
        return EmotionFusionResult(style_vector=fused, metadata=metadata)


def _normalize_scores(scores: Dict[str, float], length: int) -> np.ndarray:
    if not scores:
        return np.zeros(length, dtype=np.float32)
    sorted_items = sorted(scores.items())
    values = np.array([v for _k, v in sorted_items], dtype=np.float32)
    if values.size < length:
        padded = np.zeros(length, dtype=np.float32)
        padded[: values.size] = values
        return padded
    return values[:length]


def _prosody_to_vector(features: Dict[str, float], length: int) -> np.ndarray:
    if not features:
        return np.zeros(length, dtype=np.float32)
    items = sorted(features.items())
    values = np.array([float(v) for _k, v in items], dtype=np.float32)
    if values.size < length:
        padded = np.zeros(length, dtype=np.float32)
        padded[: values.size] = values
        return padded
    return values[:length]
