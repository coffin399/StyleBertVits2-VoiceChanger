from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from app.config import BackendConfig, ModelConfig
from vendor.sovits.hparams import get_hparams_from_file
from vendor.sovits.models import SynthesizerTrnMs256NSFsid
from vendor.sovits.utils import f0_to_coarse


@dataclass(slots=True)
class GeneratorArtifacts:
    checkpoint: Optional[Path]
    onnx: Optional[Path]
    tensorrt: Optional[Path]


class GeneratorBackend:
    """Wrapper for so-vits-svc generator supporting multiple runtimes."""

    def __init__(self, model_cfg: ModelConfig, backend_cfg: BackendConfig) -> None:
        self._model_cfg = model_cfg
        self._artifacts = GeneratorArtifacts(
            checkpoint=model_cfg.generator_checkpoint if model_cfg.generator_checkpoint.exists() else None,
            onnx=model_cfg.generator_onnx if model_cfg.generator_onnx and model_cfg.generator_onnx.exists() else None,
            tensorrt=model_cfg.tensorrt_engine if model_cfg.tensorrt_engine and model_cfg.tensorrt_engine.exists() else None,
        )
        self._backend_cfg = backend_cfg
        self._device = torch.device(backend_cfg.device)
        self._torch_model: Optional[torch.nn.Module] = None
        self._onnx_session = None
        self._tensorrt_context = None
        self._active_backend: str = "unloaded"
        self._torch_dtype: torch.dtype = torch.float32
        self._hparams = None
        self._sample_rate: Optional[int] = None
        self._hop_length: Optional[int] = None
        self._speaker_map: dict[str, int] = {}
        self._num_speakers: int = 0
        self._onnx_input_names: Sequence[str] = ()
        self._onnx_output_names: Sequence[str] = ()
        self._tensorrt_engine = None
        self._tensorrt_binding_indices: dict[str, int] = {}

    def load(self) -> None:
        """Select and load generator runtime."""

        if self._backend_cfg.prefer_tensorrt and self._artifacts.tensorrt:
            self._load_tensorrt()
            return
        if self._backend_cfg.onnx_provider and self._artifacts.onnx:
            self._load_onnx()
            return
        if self._artifacts.checkpoint:
            self._load_torch()
            return
        logger.warning("No generator artifacts found. Using dummy generator backend.")
        self._active_backend = "dummy"

    def _load_torch(self) -> None:
        logger.info("Loading PyTorch generator from {}", self._artifacts.checkpoint)
        state = torch.load(self._artifacts.checkpoint, map_location="cpu")
        if "model" in state:
            state = state["model"]
        elif "generator" in state:
            state = state["generator"]
        hps = self._load_hparams()
        spec_channels = hps.data.filter_length // 2 + 1
        segment_size = hps.train.segment_size // hps.data.hop_length
        model_kwargs = dict(vars(hps.model))
        model_kwargs.setdefault("sampling_rate", hps.data.sampling_rate)
        model = SynthesizerTrnMs256NSFsid(spec_channels, segment_size, **model_kwargs)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys while loading generator: {}", missing)
        if unexpected:
            logger.warning("Unexpected keys while loading generator: {}", unexpected)
        model.eval()
        model = model.to(self._device)
        if self._backend_cfg.use_fp16 and self._device.type == "cuda":
            model = model.half()
        self._torch_model = model
        self._torch_dtype = next(model.parameters()).dtype
        self._active_backend = "pytorch"

    def _load_onnx(self) -> None:
        import onnxruntime as ort

        providers = [self._backend_cfg.onnx_provider]
        logger.info("Loading ONNX generator {} with providers {}", self._artifacts.onnx, providers)
        self._onnx_session = ort.InferenceSession(
            str(self._artifacts.onnx),
            providers=providers,
        )
        self._onnx_input_names = tuple(inp.name for inp in self._onnx_session.get_inputs())
        self._onnx_output_names = tuple(out.name for out in self._onnx_session.get_outputs())
        self._load_hparams()
        self._active_backend = "onnx"

    def _load_tensorrt(self) -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:  # pragma: no cover - requires NVIDIA libs
            raise RuntimeError("TensorRT is not installed.") from exc

        logger.info("Loading TensorRT engine from {}", self._artifacts.tensorrt)
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        with open(self._artifacts.tensorrt, "rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")
        self._tensorrt_engine = engine
        self._tensorrt_context = engine.create_execution_context()
        self._tensorrt_binding_indices = {
            engine.get_binding_name(i): i for i in range(engine.num_bindings)
        }
        self._load_hparams()
        self._active_backend = "tensorrt"

    def infer(self, content: torch.Tensor, f0: torch.Tensor, speaker_id: int = 0) -> np.ndarray:
        """Run generator to synthesize waveform."""

        if self._tensorrt_context is not None:
            return self._infer_tensorrt(content, f0, speaker_id)
        if self._onnx_session is not None:
            return self._infer_onnx(content, f0, speaker_id)
        if self._torch_model is not None:
            return self._infer_torch(content, f0, speaker_id)
        if self._active_backend == "dummy":
            return self._infer_dummy(content)
        raise RuntimeError("Generator backend is not loaded.")

    def _infer_torch(self, content: torch.Tensor, f0: torch.Tensor, speaker_id: int) -> np.ndarray:
        if self._torch_model is None:
            raise RuntimeError("Torch generator is not loaded.")

        model = self._torch_model
        device = self._device
        dtype = self._torch_dtype

        content_cpu, f0_cpu, uv_cpu = self._prepare_base_inputs(content, f0)
        content = content_cpu.to(device=device, dtype=dtype)
        f0 = f0_cpu.to(device=device, dtype=dtype)
        uv = uv_cpu.to(device=device, dtype=dtype)
        sid_index = self._resolve_speaker_id(speaker_id)
        sid = torch.LongTensor([sid_index]).to(device).unsqueeze(0)

        with torch.no_grad():
            audio, _ = model.infer(content, f0=f0, uv=uv, g=sid, predict_f0=False, vol=None)
        waveform = audio[0, 0].float().cpu().numpy()
        return waveform

    def _infer_onnx(self, content: torch.Tensor, f0: torch.Tensor, speaker_id: int) -> np.ndarray:
        if self._onnx_session is None:
            raise RuntimeError("ONNX generator session is not initialized.")

        content_cpu, f0_cpu, uv_cpu = self._prepare_base_inputs(content, f0)
        runtime_inputs = self._build_runtime_inputs(content_cpu, f0_cpu, uv_cpu, speaker_id)

        input_feed: Dict[str, np.ndarray] = {}
        for inp in self._onnx_session.get_inputs():
            name = inp.name
            if name not in runtime_inputs:
                raise KeyError(f"ONNX input '{name}' not prepared. Available keys: {list(runtime_inputs.keys())}")
            value = runtime_inputs[name]
            value = self._reshape_numpy_to_match(value, inp.shape)
            value = self._ensure_numpy_dtype(value, inp.type)
            input_feed[name] = value

        outputs = self._onnx_session.run(None, input_feed)
        if not outputs:
            raise RuntimeError("ONNX generator returned no outputs.")
        waveform = self._extract_waveform(outputs[0])
        return waveform

    def _infer_tensorrt(self, content: torch.Tensor, f0: torch.Tensor, speaker_id: int) -> np.ndarray:
        if self._tensorrt_context is None or self._tensorrt_engine is None:
            raise RuntimeError("TensorRT context is not initialized.")
        if self._device.type != "cuda":
            raise RuntimeError("TensorRT inference requires CUDA device.")

        import tensorrt as trt

        content_cpu, f0_cpu, uv_cpu = self._prepare_base_inputs(content, f0)
        runtime_inputs = self._build_runtime_inputs(content_cpu, f0_cpu, uv_cpu, speaker_id)

        engine = self._tensorrt_engine
        context = self._tensorrt_context
        stream = torch.cuda.current_stream(device=self._device)

        bindings = [0] * engine.num_bindings
        output_tensors: Dict[str, torch.Tensor] = {}

        # Configure input bindings
        for name, index in self._tensorrt_binding_indices.items():
            if not engine.binding_is_input(index):
                continue
            if name not in runtime_inputs:
                raise KeyError(f"TensorRT binding '{name}' not prepared. Available keys: {list(runtime_inputs.keys())}")
            declared_shape = tuple(engine.get_binding_shape(index))
            value = runtime_inputs[name]
            value = self._reshape_numpy_to_target(value, declared_shape)
            value = self._ensure_numpy_dtype_for_trt(value, engine.get_binding_dtype(index))
            context.set_binding_shape(index, tuple(value.shape))
            tensor = torch.from_numpy(value).to(device=self._device)
            tensor = tensor.to(dtype=self._torch_dtype_from_numpy_dtype(trt.nptype(engine.get_binding_dtype(index))))
            bindings[index] = tensor.data_ptr()
        if not context.all_binding_shapes_specified:
            raise RuntimeError("TensorRT binding shapes are incomplete.")

        # Allocate outputs
        for name, index in self._tensorrt_binding_indices.items():
            if engine.binding_is_input(index):
                continue
            binding_shape = tuple(context.get_binding_shape(index))
            dtype = self._torch_dtype_from_numpy_dtype(trt.nptype(engine.get_binding_dtype(index)))
            tensor = torch.empty(binding_shape, dtype=dtype, device=self._device)
            output_tensors[name] = tensor
            bindings[index] = tensor.data_ptr()

        success = context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
        if not success:
            raise RuntimeError("TensorRT execute_async_v2 failed.")
        stream.synchronize()

        # Prefer first declared output
        output_name = next(iter(output_tensors))
        output_tensor = output_tensors[output_name]
        waveform = self._extract_waveform(output_tensor.detach().float().cpu().numpy())
        return waveform

    def _infer_dummy(self, content: torch.Tensor) -> np.ndarray:
        frames = content.shape[-1]
        return np.zeros(frames * 256, dtype=np.float32)

    @property
    def active_backend(self) -> str:
        return self._active_backend

    @property
    def sample_rate(self) -> Optional[int]:
        return self._sample_rate

    def _resolve_speaker_id(self, speaker: Union[int, str]) -> int:
        if isinstance(speaker, str):
            if speaker in self._speaker_map:
                return self._speaker_map[speaker]
            raise KeyError(f"Speaker '{speaker}' not found in speaker map.")
        sid = int(speaker)
        if self._num_speakers and sid >= self._num_speakers:
            logger.warning("Speaker id {} out of range; wrapping into {} speakers.", sid, self._num_speakers)
            sid %= self._num_speakers
        return max(sid, 0)

    def _load_speaker_map(self) -> dict[str, int]:
        path = self._model_cfg.speaker_map
        if not path or not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): int(v) for k, v in data.items()}
        except Exception as exc:
            logger.warning("Failed to load speaker map {}: {}", path, exc)
        return {}

    def _load_hparams(self) -> Any:
        if self._hparams is None:
            config_path = self._model_cfg.generator_config
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Generator config not found: {config_path}. Ensure so-vits-svc config.json is available."
                )
            self._hparams = get_hparams_from_file(config_path)
            self._sample_rate = int(self._hparams.data.sampling_rate)
            self._hop_length = int(self._hparams.data.hop_length)
            self._num_speakers = getattr(self._hparams.model, "n_speakers", 0) or 0
        self._speaker_map = self._load_speaker_map()
        return self._hparams

    def _prepare_base_inputs(self, content: Union[torch.Tensor, np.ndarray], f0: Union[torch.Tensor, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(content, torch.Tensor):
            content = torch.as_tensor(content)
        if not isinstance(f0, torch.Tensor):
            f0 = torch.as_tensor(f0)

        content = content.detach().to(device="cpu", dtype=torch.float32)
        f0 = f0.detach().to(device="cpu", dtype=torch.float32)

        if content.dim() == 1:
            content = content.unsqueeze(0).unsqueeze(0)
        elif content.dim() == 2:
            content = content.unsqueeze(0)
        elif content.dim() != 3:
            raise ValueError(f"Unsupported content tensor shape: {tuple(content.shape)}")

        if f0.dim() == 1:
            f0 = f0.unsqueeze(0)
        elif f0.dim() != 2:
            raise ValueError(f"Unsupported f0 tensor shape: {tuple(f0.shape)}")

        target_frames = f0.shape[-1]
        if content.shape[-1] != target_frames:
            content = F.interpolate(content, size=target_frames, mode="nearest")

        uv = (f0 > 0).to(dtype=torch.float32)
        return content.contiguous(), f0.contiguous(), uv.contiguous()

    def _build_runtime_inputs(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        uv: torch.Tensor,
        speaker_id: Union[int, str],
    ) -> Dict[str, np.ndarray]:
        batch = content.shape[0]
        frames = f0.shape[-1]
        sid_index = self._resolve_speaker_id(speaker_id)

        c_np = np.ascontiguousarray(content.numpy())
        f0_np = np.ascontiguousarray(f0.numpy())
        uv_np = np.ascontiguousarray(uv.numpy())
        sid_np = np.full((batch,), sid_index, dtype=np.int64)
        sid_col_np = sid_np[:, None]
        length_np = np.full((batch,), frames, dtype=np.int64)
        mel2ph_np = np.tile(np.arange(1, frames + 1, dtype=np.int64)[None, :], (batch, 1))
        noise_np = np.zeros((batch, 1, frames), dtype=np.float32)
        noise_flat_np = np.zeros((batch, frames), dtype=np.float32)
        vol_np = np.ones((batch, frames), dtype=np.float32)
        coarse_np = np.ascontiguousarray(f0_to_coarse(f0).numpy())

        inputs: Dict[str, np.ndarray] = {
            "c": c_np,
            "content": c_np,
            "hidden_units": c_np,
            "hubert": c_np,
            "f0": f0_np,
            "pitch": f0_np,
            "pitch_input": f0_np,
            "uv": uv_np,
            "uv_input": uv_np,
            "voiced_mask": uv_np,
            "sid": sid_np,
            "speaker": sid_np,
            "speaker_id": sid_np,
            "spk": sid_np,
            "spk_id": sid_np,
            "g": sid_col_np,
            "sid_expanded": sid_col_np,
            "length": length_np,
            "c_lengths": length_np,
            "input_lengths": length_np,
            "n_frames": length_np,
            "mel2ph": mel2ph_np,
            "mel2phone": mel2ph_np,
            "noise": noise_np,
            "noise_flat": noise_flat_np,
            "latent_noise": noise_np,
            "vol": vol_np,
            "volume": vol_np,
            "predict_f0": np.zeros((batch,), dtype=bool),
            "length_scale": np.array([1.0], dtype=np.float32),
            "noise_scale": np.array([0.667], dtype=np.float32),
            "noise_scale_w": np.array([0.8], dtype=np.float32),
            "seed": np.array([0], dtype=np.int64),
            "f0_coarse": coarse_np,
            "pitch_coarse": coarse_np,
        }
        return inputs

    def _ensure_numpy_dtype(self, value: np.ndarray, onnx_type: str) -> np.ndarray:
        type_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(bool)": np.bool_,
        }
        target = type_map.get(onnx_type)
        if target is None:
            return value
        if value.dtype != target:
            value = value.astype(target, copy=False)
        return value

    def _ensure_numpy_dtype_for_trt(self, value: np.ndarray, dtype: "trt.DataType") -> np.ndarray:
        import tensorrt as trt

        target = trt.nptype(dtype)
        if value.dtype != target:
            value = value.astype(target, copy=False)
        return value

    def _reshape_numpy_to_match(self, value: np.ndarray, target_shape: Sequence[Union[int, str]]) -> np.ndarray:
        if not target_shape:
            return value
        if sum(dim is not None and dim != -1 and dim != "None" for dim in target_shape) == 0:
            return value
        cleaned_shape: list[int] = []
        for dim in target_shape:
            if isinstance(dim, int) and dim > 0:
                cleaned_shape.append(dim)
            else:
                cleaned_shape.append(-1)
        if len(cleaned_shape) == value.ndim:
            return value
        while len(cleaned_shape) > len(value.shape):
            value = np.expand_dims(value, axis=-1)
        return value

    def _reshape_numpy_to_target(self, value: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
        if not target_shape or all(dim < 0 for dim in target_shape):
            return value
        if tuple(target_shape) == value.shape:
            return value
        element_count = np.prod(value.shape)
        target_count = np.prod(target_shape)
        if element_count != target_count:
            return value
        return value.reshape(target_shape)

    def _extract_waveform(self, output: Union[np.ndarray, Sequence[Any]]) -> np.ndarray:
        if isinstance(output, (list, tuple)):
            if not output:
                raise ValueError("Generator output is empty.")
            output = output[0]
        waveform = np.asarray(output)
        if waveform.ndim == 3:
            waveform = waveform[0, 0]
        elif waveform.ndim == 2:
            waveform = waveform[0]
        waveform = waveform.astype(np.float32, copy=False)
        return waveform

    @staticmethod
    def _torch_dtype_from_numpy_dtype(dtype: np.dtype) -> torch.dtype:
        if dtype == np.float16:
            return torch.float16
        if dtype == np.float32:
            return torch.float32
        if dtype == np.int32:
            return torch.int32
        if dtype == np.int64:
            return torch.int64
        if dtype == np.bool_:
            return torch.bool
        raise TypeError(f"Unsupported numpy dtype for TensorRT binding: {dtype}")
