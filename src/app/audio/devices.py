from __future__ import annotations

import sounddevice as sd
from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class AudioDevice:
    name: str
    index: int
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


def list_input_devices() -> List[AudioDevice]:
    devices = []
    for idx, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            devices.append(
                AudioDevice(
                    name=info["name"],
                    index=idx,
                    max_input_channels=info["max_input_channels"],
                    max_output_channels=info["max_output_channels"],
                    default_samplerate=info["default_samplerate"],
                )
            )
    return devices


def list_output_devices() -> List[AudioDevice]:
    devices = []
    for idx, info in enumerate(sd.query_devices()):
        if info["max_output_channels"] > 0:
            devices.append(
                AudioDevice(
                    name=info["name"],
                    index=idx,
                    max_input_channels=info["max_input_channels"],
                    max_output_channels=info["max_output_channels"],
                    default_samplerate=info["default_samplerate"],
                )
            )
    return devices
