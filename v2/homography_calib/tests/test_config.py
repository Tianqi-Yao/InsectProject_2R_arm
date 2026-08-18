import json

import pytest

from homography_calib import config as cfg


def test_default_calib_is_valid():
    calib = cfg.WorkspaceCalib()
    cfg.validate(calib)  # must not raise
    assert calib.H is None


def test_corner_world_points_matches_tag_id_order():
    calib = cfg.WorkspaceCalib()
    points = calib.corner_world_points()
    assert points == [(0.0, 150.0), (200.0, 150.0), (200.0, 0.0), (0.0, 0.0)]


def test_round_trip_save_load(tmp_path):
    path = tmp_path / "workspace_calib.json"
    calib = cfg.WorkspaceCalib(H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], reproj_rms_px=0.5)
    cfg.save(calib, path)
    loaded = cfg.load(path)
    assert loaded == calib


def test_load_missing_file_returns_defaults(tmp_path):
    calib = cfg.load(tmp_path / "nope.json")
    assert calib == cfg.WorkspaceCalib()


def test_load_tolerates_unknown_keys(tmp_path):
    path = tmp_path / "workspace_calib.json"
    path.write_text(json.dumps({"width_mm": 999.0, "unknown_future_field": "x"}))
    calib = cfg.load(path)
    assert calib.width_mm == 999.0


def test_save_backs_up_previous_file(tmp_path):
    path = tmp_path / "workspace_calib.json"
    cfg.save(cfg.WorkspaceCalib(width_mm=100.0), path)
    cfg.save(cfg.WorkspaceCalib(width_mm=200.0), path)
    backups = list((tmp_path / cfg.HISTORY_DIRNAME).glob("*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text())["width_mm"] == 100.0


def test_validate_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="width_mm"):
        cfg.validate(cfg.WorkspaceCalib(width_mm=0.0))


def test_validate_rejects_missing_corner_keys():
    calib = cfg.WorkspaceCalib(corner_tag_ids={"tl": 0, "tr": 1, "br": 2})
    with pytest.raises(ValueError, match="corner_tag_ids"):
        cfg.validate(calib)


def test_validate_rejects_corner_tag_without_world_coords():
    calib = cfg.WorkspaceCalib(corner_tag_ids={"tl": 99, "tr": 1, "br": 2, "bl": 3})
    with pytest.raises(ValueError, match="tl"):
        cfg.validate(calib)


def test_validate_rejects_malformed_h_matrix():
    calib = cfg.WorkspaceCalib(H=[[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="3x3"):
        cfg.validate(calib)


def test_load_and_save_honor_a_monkeypatched_default_path(tmp_path, monkeypatch):
    """Regression test: see hw_core's equivalent test -- a `path: Path =
    DEFAULT_PATH` parameter default binds once at import time, so
    monkeypatching cfg.DEFAULT_PATH must still be honored by calls that
    don't pass an explicit path."""
    fake_default = tmp_path / "workspace_calib.json"
    monkeypatch.setattr(cfg, "DEFAULT_PATH", fake_default)
    cfg.save(cfg.WorkspaceCalib(width_mm=321.0))
    assert fake_default.exists()
    assert cfg.load().width_mm == 321.0
