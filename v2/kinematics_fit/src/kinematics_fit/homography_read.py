"""Read-only consumer of homography_calib's workspace_calib.json output.

Deliberately reimplements apply_homography (a few lines of cv2 math)
rather than importing homography_calib -- v2's cross-package rule is that
a JSON file is a shared convention, code is not. See homography_calib's
own homography.py for the twin of this function."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Matches homography_calib.config.DEFAULT_PATH exactly -- this is the
# cross-package file-format convention, not a code dependency.
WORKSPACE_CALIB_PATH = Path.home() / ".config" / "arm2r" / "workspace_calib.json"


def load_homography(path: Optional[Path] = None) -> np.ndarray:
    path = path if path is not None else WORKSPACE_CALIB_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `arm-homography fit` first")
    with open(path) as f:
        data = json.load(f)
    H = data.get("H")
    if H is None:
        raise ValueError(f"{path} exists but has no fitted homography yet -- "
                          f"run `arm-homography fit`")
    return np.array(H)


def apply_homography(H: np.ndarray, pixel_xy: tuple[float, float]) -> tuple[float, float]:
    pt = np.array([[pixel_xy]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])
