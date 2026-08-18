"""Persistence for physical-rig-level facts shared by every feature package:
which serial port the servo bridge is on, which bus IDs are joint1/joint2,
which camera backend to use, and -- the safety-critical part -- the
software joint-limit ranges written by `arm-hw set-joint-limits`.

This is deliberately the ONLY state file every feature package reads (they
each own their own separate config file for anything mode-specific -- see
each package's own config.py). Keeping hw_state.json's shape small and
stable is what lets five independently-implemented feature packages agree
on how to talk to the hardware without sharing any code beyond this
package.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path.home() / ".config" / "arm2r" / "hw_state.json"
HISTORY_DIRNAME = "hw_state_history"


@dataclass
class HwState:
    servo_port: str = "/dev/cu.usbserial-0001"
    joint_ids: dict[str, int] = field(default_factory=lambda: {"joint1": 1, "joint2": 2})
    camera_backend: str = "usb"  # "usb" | "picamera2"
    camera_resolution: tuple[int, int] = (1920, 1080)
    usb_camera_index: int = 0
    # None = fresh install, unconfigured -- within_joint_limits() then
    # allows everything. Populate via `arm-hw set-joint-limits`.
    joint_limits_deg: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HwState":
        # Tolerant load: unknown/stale keys are dropped, missing keys fall
        # back to the dataclass defaults above -- so an older hw_state.json
        # (or one hand-edited to add a field this version doesn't know
        # about yet) doesn't hard-fail a load.
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known}
        state = cls(**filtered)
        if state.camera_resolution is not None:
            state.camera_resolution = tuple(state.camera_resolution)
        return state


def validate(state: HwState) -> None:
    """Raises ValueError on anything that would make within_joint_limits or
    servos.set_hardware_angle_limits behave surprisingly. Deliberately
    strict: a partially-configured joint_limits_deg is an error, not a
    silent fallback to "unrestricted"."""
    if state.camera_backend not in ("usb", "picamera2"):
        raise ValueError(f"camera_backend must be 'usb' or 'picamera2', got {state.camera_backend!r}")
    if set(state.joint_ids) != {"joint1", "joint2"}:
        raise ValueError(f"joint_ids must have exactly joint1/joint2 keys, got {state.joint_ids!r}")

    jl = state.joint_limits_deg
    if jl is None:
        return
    for name in ("joint1", "joint2"):
        if name not in jl:
            raise ValueError(f"joint_limits_deg is set but missing {name!r}")
        lo, hi = jl[name]
        if not (0.0 <= lo < hi <= 360.0):
            raise ValueError(f"joint_limits_deg[{name!r}] = [{lo}, {hi}] must satisfy "
                              f"0 <= lo < hi <= 360 (non-wrapping range)")
    boundary = jl.get("coupled_boundary") or []
    if boundary and len(boundary) < 3:
        raise ValueError(f"coupled_boundary has {len(boundary)} vertices, need >=3 for a polygon")
    for v in boundary:
        if len(v) != 2 or not all(isinstance(c, (int, float)) for c in v):
            raise ValueError(f"coupled_boundary vertex {v!r} must be a 2-element (joint1, joint2) pair")


def load(path: Optional[Path] = None) -> HwState:
    # `path` resolves against the CURRENT value of DEFAULT_PATH at call
    # time, not whatever it was when this function was defined -- a plain
    # `path: Path = DEFAULT_PATH` default would bind once at import time,
    # silently ignoring any later `hw_state.DEFAULT_PATH = ...` override
    # (e.g. in tests).
    path = path if path is not None else DEFAULT_PATH
    if not path.exists():
        return HwState()
    with open(path) as f:
        return HwState.from_dict(json.load(f))


def save(state: HwState, path: Optional[Path] = None) -> None:
    """Validates before writing, and backs up whatever was there before
    overwriting it -- a manual-rollback safety net, not a curated history
    (nothing reads these snapshots back automatically)."""
    path = path if path is not None else DEFAULT_PATH
    validate(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history_dir = path.parent / HISTORY_DIRNAME
        history_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        shutil.copy(path, history_dir / f"hw_state_{stamp}.json")
    with open(path, "w") as f:
        json.dump(state.to_dict(), f, indent=2)
        f.write("\n")
