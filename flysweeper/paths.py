from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FLYSWEEPER_DATA", ROOT / "data"))
RAW = DATA / "raw"
COMPILED = DATA / "compiled"
OUTPUTS = Path(os.environ.get("FLYSWEEPER_OUTPUTS", ROOT / "outputs"))


def ensure_dirs() -> None:
    for d in (RAW, COMPILED, OUTPUTS):
        d.mkdir(parents=True, exist_ok=True)
