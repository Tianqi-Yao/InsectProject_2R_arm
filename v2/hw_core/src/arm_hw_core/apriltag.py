"""Thin wrapper around pupil_apriltags.Detector, keyed by tag_id for easy
lookup by every feature package that needs to find a specific tag."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    tag_id: int
    center: tuple[float, float]
    corners: list[tuple[float, float]]


class TagDetector:
    def __init__(self, family: str = "tag36h11"):
        from pupil_apriltags import Detector  # deferred import

        self._detector = Detector(families=family, nthreads=2, quad_decimate=1.0)

    def detect(self, frame) -> dict[int, Detection]:
        results = self._detector.detect(frame)
        return {
            r.tag_id: Detection(tag_id=r.tag_id, center=tuple(r.center), corners=list(r.corners))
            for r in results
        }
