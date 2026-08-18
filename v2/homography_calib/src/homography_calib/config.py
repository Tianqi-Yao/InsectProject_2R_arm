"""Persistence for workspace_calib.json: the 4 corner AprilTags' known
world coordinates and the fitted pixel<->mm homography. This is the file
kinematics_fit and card_scan read (read-only) to convert pixel positions to
mm -- see this package's README for the cross-package data contract."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path.home() / ".config" / "arm2r" / "workspace_calib.json"
HISTORY_DIRNAME = "workspace_calib_history"


@dataclass
class WorkspaceCalib:
    width_mm: float = 200.0
    height_mm: float = 150.0
    corner_tag_ids: dict[str, int] = field(
        default_factory=lambda: {"tl": 0, "tr": 1, "br": 2, "bl": 3})
    # Keyed by tag_id (as a string, for JSON compatibility) -> (x_mm, y_mm).
    corner_world_mm: dict[str, list[float]] = field(default_factory=lambda: {
        "0": [0.0, 150.0], "1": [200.0, 150.0], "2": [200.0, 0.0], "3": [0.0, 0.0],
    })
    drift_halt_mm: float = 3.0
    H: Optional[list[list[float]]] = None
    computed_at: Optional[str] = None
    reproj_rms_px: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "width_mm": self.width_mm, "height_mm": self.height_mm,
            "corner_tag_ids": self.corner_tag_ids, "corner_world_mm": self.corner_world_mm,
            "drift_halt_mm": self.drift_halt_mm,
            "H": self.H, "computed_at": self.computed_at, "reproj_rms_px": self.reproj_rms_px,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkspaceCalib":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def corner_world_points(self) -> list[tuple[float, float]]:
        """(pixel-detection-order-independent) list of world (x, y) points,
        in the same tl/tr/br/bl order as corner_tag_ids -- callers zip this
        against detected pixel centers in that same order."""
        return [tuple(self.corner_world_mm[str(tag_id)])
                for tag_id in self.corner_tag_ids.values()]


def validate(calib: WorkspaceCalib) -> None:
    if calib.width_mm <= 0 or calib.height_mm <= 0:
        raise ValueError("width_mm/height_mm must be positive")
    if set(calib.corner_tag_ids) != {"tl", "tr", "br", "bl"}:
        raise ValueError(f"corner_tag_ids must have exactly tl/tr/br/bl keys, "
                          f"got {calib.corner_tag_ids!r}")
    for corner, tag_id in calib.corner_tag_ids.items():
        if str(tag_id) not in calib.corner_world_mm:
            raise ValueError(f"corner_world_mm is missing an entry for {corner}'s "
                              f"tag_id={tag_id}")
    if calib.H is not None and (len(calib.H) != 3 or any(len(row) != 3 for row in calib.H)):
        raise ValueError("H must be a 3x3 matrix")


def load(path: Optional[Path] = None) -> WorkspaceCalib:
    # Resolved against the CURRENT value of DEFAULT_PATH at call time, not
    # whatever it was when this function was defined -- see hw_core's
    # hw_state.load() for why a plain `path: Path = DEFAULT_PATH` default
    # would silently ignore a later override (e.g. in tests).
    path = path if path is not None else DEFAULT_PATH
    if not path.exists():
        return WorkspaceCalib()
    with open(path) as f:
        return WorkspaceCalib.from_dict(json.load(f))


def save(calib: WorkspaceCalib, path: Optional[Path] = None) -> None:
    path = path if path is not None else DEFAULT_PATH
    validate(calib)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history_dir = path.parent / HISTORY_DIRNAME
        history_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        shutil.copy(path, history_dir / f"workspace_calib_{stamp}.json")
    with open(path, "w") as f:
        json.dump(calib.to_dict(), f, indent=2)
        f.write("\n")
