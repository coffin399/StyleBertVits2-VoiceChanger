from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from loguru import logger
from safetensors.torch import load_file
from torch import Tensor
from transformers import AutoConfig, AutoModel

from app.config import AudioConfig, ModelConfig


@dataclass(slots=True)
class ContentEncoderConfig:
    model_dir: Path
    sample_rate: int
    device: torch.device


class JapaneseHubertContentEncoder:
    """Content encoder backed by rinna/japanese-hubert-base."""

    def __init__(
        self,
        models: ModelConfig,
        audio: AudioConfig,
        device: str = "cuda",
    ) -> None:
        content_path = models.content_encoder
        if not content_path.exists():
            raise FileNotFoundError(
                f"Content encoder checkpoint not found: {content_path}\n"
                "Download the repository from https://huggingface.co/rinna/japanese-hubert-base"
                " and place model.safetensors in the configured directory."
            )
        model_dir = content_path.parent
        self._config = ContentEncoderConfig(
            model_dir=model_dir,
            sample_rate=audio.content_sample_rate,
            device=torch.device(device),
        )
        logger.info("Loading Japanese HuBERT content encoder from {}", model_dir)
        hf_config = AutoConfig.from_pretrained(model_dir)
        model = AutoModel.from_config(hf_config)
        state_dict = load_file(str(content_path), device="cpu")
        model.load_state_dict(state_dict)
        self._model = model.to(self._config.device)
        self._model.eval()
        self._feature_extractor = torchaudio.transforms.Resample(
            orig_freq=audio.input_sample_rate,
            new_freq=self._config.sample_rate,
        ).to(self._config.device)

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor, sample_rate: int) -> Tensor:
        """Compute content embeddings for a single-channel audio tensor."""

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if sample_rate != self._config.sample_rate:
            audio = self._feature_extractor(audio)
        audio = audio.to(self._config.device)
        outputs = self._model(audio, output_hidden_states=True)
        # HuBERT last hidden state shape: (batch, frames, dim)
        hidden: Tensor = outputs.last_hidden_state
        return hidden.transpose(1, 2).contiguous()

    @property
    def device(self) -> torch.device:
        return self._config.device
