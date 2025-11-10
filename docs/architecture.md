# System Architecture Overview

## Goals
- Real-time voice conversion pipeline using Whisper, emotion analysis (BERT + openSMILE), and StyleBertVits2.
- Provide GUI with input/output device selection and local playback output.
- Vendor StyleBertVits2 core inference components while minimizing bundled files and preserving licensing.

## High-Level Pipeline
1. **Audio Capture** (`sounddevice`): stream microphone input.
2. **Speech-to-Text** (`faster-whisper`): transcribe buffered audio segments.
3. **Emotion Analysis**:
   - **Text Emotion (BERT)**: Transformers Japanese BERT fine-tuned (e.g., `daigo/bert-base-japanese-emotion`).
   - **Prosody Features (openSMILE)**: Extract pitch, energy, tempo features from the same buffered audio.
4. **Emotion Fusion**: Combine text emotion distribution and prosodic metrics into Style conditioning vector (weighted rules with configurable parameters).
5. **StyleBertVits2 Synthesis**: Generate waveform using fused emotions and recognized text.
6. **Output Routing**:
   - **System Playback**: `sounddevice` playback queue.
   - **File Export**: optional WAV saving (planned).

## Module Layout (`src/`)
- `app/gui/`: Qt main window, device selection, status monitors.
- `app/audio/`: input capture, buffering, playback routing.
- `app/transcription/`: Whisper wrapper with async queue.
- `app/emotion/`: text emotion scorer and prosody extractor.
- `app/style/`: fusion logic and StyleBertVits2 bridge.
- `app/output/`: system audio sink, (future) file writer.
- `app/config.py`: user-configurable parameters, model paths.

## StyleBertVits2 Extraction Strategy
- Retain only required modules for inference:
  - `style_bert_vits2/tts_model.py`
  - `style_bert_vits2/constants.py`
  - `style_bert_vits2/logging.py`
  - `style_bert_vits2/voice.py`
  - `style_bert_vits2/models/` subset: `__init__`, `infer.py`, ONNX infer (optional), hyper parameters, text pipeline dependencies.
  - `style_bert_vits2/utils/` minimal helpers used by retained modules.
- Remove training scripts, datasets, Gradio app, tests, Docker files.
- Copy required assets (e.g., default config, style vectors if needed) into `vendor/style_bert_vits2/`.
- Keep `LICENSE`, `LGPL_LICENSE`, and add NOTICE describing extractions and modifications.

## Dependency Notes
- Ensure PyTorch build compatible with GPU (user to install appropriate wheel).
- `opensmile` Python bindings may require prebuilt binaries; provide fallback stub with warning if unavailable.
- Provide configuration for ONNX vs. PyTorch inference; default to PyTorch `.safetensors` model.

## Next Steps
1. Script to copy minimal StyleBertVits2 files into `vendor/` and delete unused items.
2. Define configuration dataclasses and module scaffolding under `src/app/`.
3. Implement audio capture/transcription pipeline with async queues.
4. Integrate style fusion and synthesis.
5. Build Qt GUI and system playback integration.
6. Update documentation & NOTICE.
