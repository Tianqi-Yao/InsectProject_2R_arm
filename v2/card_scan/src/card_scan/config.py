"""Persistence for card_scan.json: the card's own tag id + physical size
(hand-measured -- can't be derived from the tag, which is much smaller
than the card), grid density, and the manual-corner fallback for when the
card's tag can't be reliably auto-detected."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path.home() / ".config" / "arm2r" / "card_scan.json"
HISTORY_DIRNAME = "card_scan_history"


@dataclass
class CardScanConfig:
    tag_id: Optional[int] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    rows: int = 3
    cols: int = 3
    dwell_s: float = 1.0
    # Fallback scan-area rectangle (world mm) when live tag detection isn't
    # trusted -- both must be set together, or neither (see validate()).
    manual_corner_a_mm: Optional[list[float]] = None
    manual_corner_b_mm: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return {
            "tag_id": self.tag_id, "width_mm": self.width_mm, "height_mm": self.height_mm,
            "rows": self.rows, "cols": self.cols, "dwell_s": self.dwell_s,
            "manual_corner_a_mm": self.manual_corner_a_mm,
            "manual_corner_b_mm": self.manual_corner_b_mm,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CardScanConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def validate(calib: CardScanConfig) -> None:
    if calib.rows < 2 or calib.cols < 2:
        raise ValueError(f"rows and cols must both be >=2, got rows={calib.rows}, cols={calib.cols}")
    if calib.dwell_s <= 0:
        raise ValueError("dwell_s must be positive")
    if (calib.tag_id is not None) != (calib.width_mm is not None and calib.height_mm is not None):
        raise ValueError("tag_id and width_mm/height_mm must be configured together, "
                          "or not at all -- a partially-configured card is an error")
    if (calib.manual_corner_a_mm is None) != (calib.manual_corner_b_mm is None):
        raise ValueError("manual_corner_a_mm and manual_corner_b_mm must be set together, "
                          "or not at all")


def load(path: Optional[Path] = None) -> CardScanConfig:
    path = path if path is not None else DEFAULT_PATH
    if not path.exists():
        return CardScanConfig()
    with open(path) as f:
        return CardScanConfig.from_dict(json.load(f))


def save(calib: CardScanConfig, path: Optional[Path] = None) -> None:
    path = path if path is not None else DEFAULT_PATH
    validate(calib)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history_dir = path.parent / HISTORY_DIRNAME
        history_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        shutil.copy(path, history_dir / f"card_scan_{stamp}.json")
    with open(path, "w") as f:
        json.dump(calib.to_dict(), f, indent=2)
        f.write("\n")
