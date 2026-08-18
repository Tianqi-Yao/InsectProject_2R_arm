"""Persistence for recorded_path.json: an ordered list of hand-recorded
(joint1_deg, joint2_deg) points, plus replay tuning (dwell/photo timing)."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path.home() / ".config" / "arm2r" / "recorded_path.json"
HISTORY_DIRNAME = "recorded_path_history"


@dataclass
class RecordedPoint:
    joint1_deg: float
    joint2_deg: float


@dataclass
class RecordedPath:
    points: list[RecordedPoint] = field(default_factory=list)
    dwell_s: float = 5.0
    # Seconds into the dwell before the shutter fires -- gives the arm's
    # own vibration time to settle after the controller reports
    # is_moving == False (that flag means "trajectory finished," not
    # "perfectly still"). Must be < dwell_s.
    photo_delay_s: float = 3.0

    def to_dict(self) -> dict:
        return {
            "points": [[p.joint1_deg, p.joint2_deg] for p in self.points],
            "dwell_s": self.dwell_s,
            "photo_delay_s": self.photo_delay_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RecordedPath":
        points = [RecordedPoint(j1, j2) for j1, j2 in d.get("points", [])]
        return cls(points=points,
                    dwell_s=d.get("dwell_s", 5.0),
                    photo_delay_s=d.get("photo_delay_s", 3.0))


def validate(path_cfg: RecordedPath) -> None:
    if path_cfg.dwell_s <= 0:
        raise ValueError("dwell_s must be positive")
    if not (0.0 < path_cfg.photo_delay_s < path_cfg.dwell_s):
        raise ValueError(f"photo_delay_s ({path_cfg.photo_delay_s}) must be between 0 and "
                          f"dwell_s ({path_cfg.dwell_s}) for a photo to fit inside the dwell")


def load(path: Optional[Path] = None) -> RecordedPath:
    path = path if path is not None else DEFAULT_PATH
    if not path.exists():
        return RecordedPath()
    with open(path) as f:
        return RecordedPath.from_dict(json.load(f))


def save(path_cfg: RecordedPath, path: Optional[Path] = None) -> None:
    path = path if path is not None else DEFAULT_PATH
    validate(path_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history_dir = path.parent / HISTORY_DIRNAME
        history_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        shutil.copy(path, history_dir / f"recorded_path_{stamp}.json")
    with open(path, "w") as f:
        json.dump(path_cfg.to_dict(), f, indent=2)
        f.write("\n")
