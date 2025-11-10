from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtWidgets

from app.config import AppConfig
from app.gui import VoiceChangerWindow


def main() -> int:
    base_dir = Path(__file__).resolve().parent.parent
    config = AppConfig()
    config.ensure_directories()

    app = QtWidgets.QApplication(sys.argv)
    window = VoiceChangerWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
