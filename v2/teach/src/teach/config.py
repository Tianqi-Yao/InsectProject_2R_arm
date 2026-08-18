"""Persistence for teach.json: named, hand-taught (joint1_deg, joint2_deg)
points -- raw servo angles are the source of truth for replay; the
workspace-mm coordinates are stored alongside purely for human-readable
display (see kinematics_read.py)."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path.home() / ".config" / "arm2r" / "teach.json"
HISTORY_DIRNAME = "teach_history"


@dataclass
class TaughtPoint:
    label: str
    joint1_deg: float
    joint2_deg: float
    x_mm: Optional[float] = None
    y_mm: Optional[float] = None


@dataclass
class TeachConfig:
    points: list[TaughtPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"points": [p.__dict__ for p in self.points]}

    @classmethod
    def from_dict(cls, d: dict) -> "TeachConfig":
        return cls(points=[TaughtPoint(**p) for p in d.get("points", [])])

    def get(self, label: str) -> Optional[TaughtPoint]:
        return next((p for p in self.points if p.label == label), None)

    def upsert(self, point: TaughtPoint) -> None:
        """Add a new point, or overwrite the existing one with the same
        label -- re-teaching a label is expected to replace it, not
        accumulate duplicates."""
        existing = self.get(point.label)
        if existing is not None:
            self.points.remove(existing)
        self.points.append(point)

    def remove(self, label: str) -> bool:
        existing = self.get(label)
        if existing is None:
            return False
        self.points.remove(existing)
        return True


def validate(calib: TeachConfig) -> None:
    labels = [p.label for p in calib.points]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate point labels: {labels!r}")
    for p in calib.points:
        if not p.label:
            raise ValueError("a taught point's label must be non-empty")


def load(path: Optional[Path] = None) -> TeachConfig:
    path = path if path is not None else DEFAULT_PATH
    if not path.exists():
        return TeachConfig()
    with open(path) as f:
        return TeachConfig.from_dict(json.load(f))


def save(calib: TeachConfig, path: Optional[Path] = None) -> None:
    path = path if path is not None else DEFAULT_PATH
    validate(calib)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history_dir = path.parent / HISTORY_DIRNAME
        history_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        shutil.copy(path, history_dir / f"teach_{stamp}.json")
    with open(path, "w") as f:
        json.dump(calib.to_dict(), f, indent=2)
        f.write("\n")
