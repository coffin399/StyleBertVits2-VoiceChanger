from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from transformers import pipeline

from app.config import EmotionConfig


@dataclass(slots=True)
class TextEmotionResult:
    label_scores: Dict[str, float]


class TextEmotionAnalyzer:
    """Wrap a transformers text-classification pipeline for emotion inference."""

    def __init__(self, config: EmotionConfig) -> None:
        self._pipeline = pipeline(
            task="text-classification",
            model=config.text_model_name,
            top_k=None,
            device_map="auto" if config.device == "auto" else None,
            device=config.device if config.device != "auto" else None,
        )

    def analyze(self, text: str) -> TextEmotionResult:
        if not text.strip():
            return TextEmotionResult(label_scores={})
        scores = self._pipeline(text)[0]
        label_scores = {entry["label"].lower(): float(entry["score"]) for entry in scores}
        total = sum(label_scores.values())
        if total > 0:
            label_scores = {k: v / total for k, v in label_scores.items()}
        return TextEmotionResult(label_scores=label_scores)
