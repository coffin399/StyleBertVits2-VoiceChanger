from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


def _to_attrdict(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return SimpleNamespace(**{k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(item) for item in obj]
    return obj


def get_hparams_from_file(config_path: Path | str) -> SimpleNamespace:
    """Load so-vits-svc config.json into an attribute-accessible namespace."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _to_attrdict(data)
