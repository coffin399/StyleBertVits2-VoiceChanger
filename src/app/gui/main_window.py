from __future__ import annotations

import asyncio
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from app.audio.devices import list_input_devices, list_output_devices
from app.config import AppConfig
from app.pipeline.voice_pipeline import VoiceProcessingPipeline


class VoiceChangerWindow(QtWidgets.QMainWindow):
    def __init__(self, config: AppConfig, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("StyleBertVits2 Voice Changer")
        self.resize(960, 640)

        self._config = config
        self._pipeline = VoiceProcessingPipeline(config)
        self._init_ui()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._refresh_state)

    def _init_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        device_group = QtWidgets.QGroupBox("デバイス設定")
        device_layout = QtWidgets.QFormLayout(device_group)
        self._input_combo = QtWidgets.QComboBox()
        self._output_combo = QtWidgets.QComboBox()
        for device in list_input_devices():
            self._input_combo.addItem(f"{device.name} ({device.index})", device.index)
        for device in list_output_devices():
            self._output_combo.addItem(f"{device.name} ({device.index})", device.index)
        device_layout.addRow("入力デバイス", self._input_combo)
        device_layout.addRow("出力デバイス", self._output_combo)
        layout.addWidget(device_group)

        control_layout = QtWidgets.QHBoxLayout()
        self._start_button = QtWidgets.QPushButton("開始")
        self._stop_button = QtWidgets.QPushButton("停止")
        self._stop_button.setEnabled(False)
        control_layout.addWidget(self._start_button)
        control_layout.addWidget(self._stop_button)
        layout.addLayout(control_layout)

        self._transcript_label = QtWidgets.QLabel("認識テキスト: ")
        self._emotion_label = QtWidgets.QLabel("感情: ")
        self._prosody_label = QtWidgets.QLabel("韻律特徴: ")
        for label in [self._transcript_label, self._emotion_label, self._prosody_label]:
            label.setWordWrap(True)
        layout.addWidget(self._transcript_label)
        layout.addWidget(self._emotion_label)
        layout.addWidget(self._prosody_label)

        self._status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self._status_bar)
        self.setCentralWidget(central)

        self._start_button.clicked.connect(self._start_pipeline)
        self._stop_button.clicked.connect(self._stop_pipeline)

    def _start_pipeline(self) -> None:
        self._config.input_device_index = self._input_combo.currentData()
        self._config.output_device_index = self._output_combo.currentData()
        self._pipeline.start()
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._timer.start()

    def _stop_pipeline(self) -> None:
        self._timer.stop()
        self._pipeline.stop()
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)

    def _refresh_state(self) -> None:
        state = self._pipeline.state
        self._transcript_label.setText(f"認識テキスト: {state.last_transcript}")
        if state.last_emotion and state.last_emotion.label_scores:
            emotion_str = ", ".join(
                f"{label}: {score:.2f}" for label, score in state.last_emotion.label_scores.items()
            )
        else:
            emotion_str = "なし"
        self._emotion_label.setText(f"感情: {emotion_str}")

        if state.last_prosody and state.last_prosody.features:
            prosody_str = ", ".join(
                f"{name}: {value:.2f}" for name, value in list(state.last_prosody.features.items())[:6]
            )
        else:
            prosody_str = "なし"
        self._prosody_label.setText(f"韻律特徴: {prosody_str}")

        if state.last_fusion_metadata:
            self._status_bar.showMessage(
                ", ".join(
                    f"{key}: {value:.2f}" for key, value in list(state.last_fusion_metadata.items())[:6]
                )
            )
        else:
            self._status_bar.clearMessage()
