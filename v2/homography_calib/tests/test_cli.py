import numpy as np
import pytest
from arm_hw_core.apriltag import Detection

from homography_calib import cli, config as cfg
from homography_calib.homography import compute_homography

PIXELS = {0: (0.0, 100.0), 1: (100.0, 100.0), 2: (100.0, 0.0), 3: (0.0, 0.0)}


class _FakeCamera:
    def connect(self):
        pass

    def close(self):
        pass

    def capture_gray(self):
        return "frame"


class _FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


def _install_fakes(monkeypatch, detections):
    monkeypatch.setattr(cli, "build_camera", lambda *a, **k: _FakeCamera())
    monkeypatch.setattr(cli, "TagDetector", lambda: _FakeDetector(detections))


def _default_detections():
    return {
        tag_id: Detection(tag_id=tag_id, center=px, corners=[])
        for tag_id, px in PIXELS.items()
    }


def test_fit_saves_homography(tmp_path, monkeypatch):
    calib_path = tmp_path / "workspace_calib.json"
    monkeypatch.setattr(cfg, "DEFAULT_PATH", calib_path)
    _install_fakes(monkeypatch, _default_detections())

    cli.cmd_fit(argparse_namespace())

    saved = cfg.load(calib_path)
    assert saved.H is not None
    assert saved.reproj_rms_px == pytest.approx(0.0, abs=1e-6)


def test_fit_raises_if_a_corner_tag_is_missing(tmp_path, monkeypatch):
    calib_path = tmp_path / "workspace_calib.json"
    monkeypatch.setattr(cfg, "DEFAULT_PATH", calib_path)
    detections = _default_detections()
    del detections[2]  # 'br' corner never detected
    _install_fakes(monkeypatch, detections)

    with pytest.raises(RuntimeError, match="br"):
        cli.cmd_fit(argparse_namespace())


def test_selfcheck_exits_1_when_no_homography_fitted(tmp_path, monkeypatch, capsys):
    calib_path = tmp_path / "workspace_calib.json"
    monkeypatch.setattr(cfg, "DEFAULT_PATH", calib_path)
    _install_fakes(monkeypatch, _default_detections())

    with pytest.raises(SystemExit) as exc:
        cli.cmd_selfcheck(argparse_namespace())
    assert exc.value.code == 1


def test_selfcheck_self_heals_when_drift_below_threshold(tmp_path, monkeypatch, capsys):
    calib_path = tmp_path / "workspace_calib.json"
    monkeypatch.setattr(cfg, "DEFAULT_PATH", calib_path)
    calib = cfg.WorkspaceCalib()
    H, _ = compute_homography(list(PIXELS.values()), calib.corner_world_points())
    calib.H = H.tolist()
    cfg.save(calib, calib_path)

    _install_fakes(monkeypatch, _default_detections())  # identical pixels -> zero drift
    cli.cmd_selfcheck(argparse_namespace())

    out = capsys.readouterr().out
    assert "OK" in out
    assert cfg.load(calib_path).H is not None


def test_selfcheck_halts_when_drift_at_or_above_threshold(tmp_path, monkeypatch):
    calib_path = tmp_path / "workspace_calib.json"
    monkeypatch.setattr(cfg, "DEFAULT_PATH", calib_path)
    calib = cfg.WorkspaceCalib(drift_halt_mm=3.0)
    H, _ = compute_homography(list(PIXELS.values()), calib.corner_world_points())
    calib.H = H.tolist()
    cfg.save(calib, calib_path)

    shifted = _default_detections()
    # World spans 0..200mm over a 0..100px square -> 1px shift = 2mm; shift
    # corner 0 by 2px so the induced drift (4mm) clears the 3mm threshold.
    shifted[0] = Detection(tag_id=0, center=(2.0, 100.0), corners=[])
    _install_fakes(monkeypatch, shifted)

    with pytest.raises(SystemExit) as exc:
        cli.cmd_selfcheck(argparse_namespace())
    assert exc.value.code == 1


def argparse_namespace():
    import argparse
    return argparse.Namespace()
