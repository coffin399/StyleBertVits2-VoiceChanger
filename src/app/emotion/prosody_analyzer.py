from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import librosa

try:
    import opensmile  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    opensmile = None


def _compute_basic_features(samples: np.ndarray, samplerate: int) -> Dict[str, float]:
    """Fallback prosody feature extraction using librosa."""

    feats: Dict[str, float] = {}
    if samples.size == 0:
        return feats
    pitches, magnitudes = librosa.piptrack(y=samples, sr=samplerate)
    pitch = pitches[magnitudes > np.median(magnitudes)]
    if pitch.size:
        feats["pitch_mean"] = float(np.mean(pitch))
        feats["pitch_std"] = float(np.std(pitch))
    rms = librosa.feature.rms(y=samples)
    if rms.size:
        feats["energy_mean"] = float(np.mean(rms))
        feats["energy_std"] = float(np.std(rms))
    tempo, _ = librosa.beat.beat_track(y=samples, sr=samplerate)
    feats["tempo"] = float(tempo)
    return feats


@dataclass(slots=True)
class ProsodyResult:
    features: Dict[str, float]


class ProsodyAnalyzer:
    """Extract prosodic descriptors using openSMILE when available, otherwise librosa."""

    def __init__(self, config_name: str, feature_level: str) -> None:
        self._use_opensmile = False
        if opensmile is not None:
            try:
                self._smile = opensmile.Smile(
                    feature_set=getattr(opensmile.FeatureSet, config_name),
                    feature_level=getattr(opensmile.FeatureLevel, feature_level),
                )
                self._use_opensmile = True
            except Exception:
                self._smile = None
        else:
            self._smile = None

    def analyze(self, samples: np.ndarray, samplerate: int) -> ProsodyResult:
        if samples.size == 0:
            return ProsodyResult(features={})
        if self._use_opensmile and self._smile is not None:
            df = self._smile.process_signal(samples, samplerate)
            features = {col: float(df[col].iloc[0]) for col in df.columns}
            return ProsodyResult(features=features)
        return ProsodyResult(features=_compute_basic_features(samples, samplerate))
