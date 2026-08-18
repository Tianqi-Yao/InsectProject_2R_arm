import pytest

from record_replay import config as cfg


def test_default_path_is_empty_and_valid():
    path_cfg = cfg.RecordedPath()
    cfg.validate(path_cfg)
    assert path_cfg.points == []


def test_round_trip_save_load(tmp_path):
    path = tmp_path / "recorded_path.json"
    path_cfg = cfg.RecordedPath(
        points=[cfg.RecordedPoint(10.0, 20.0), cfg.RecordedPoint(30.0, 40.0)],
        dwell_s=4.0, photo_delay_s=1.5)
    cfg.save(path_cfg, path)
    loaded = cfg.load(path)
    assert loaded == path_cfg


def test_load_missing_file_returns_defaults(tmp_path):
    assert cfg.load(tmp_path / "nope.json") == cfg.RecordedPath()


def test_validate_rejects_non_positive_dwell():
    with pytest.raises(ValueError, match="dwell_s"):
        cfg.validate(cfg.RecordedPath(dwell_s=0.0))


def test_validate_rejects_photo_delay_outside_dwell_window():
    with pytest.raises(ValueError, match="photo_delay_s"):
        cfg.validate(cfg.RecordedPath(dwell_s=3.0, photo_delay_s=5.0))


def test_save_backs_up_previous_file(tmp_path):
    path = tmp_path / "recorded_path.json"
    cfg.save(cfg.RecordedPath(points=[cfg.RecordedPoint(1.0, 1.0)]), path)
    cfg.save(cfg.RecordedPath(points=[cfg.RecordedPoint(2.0, 2.0)]), path)
    backups = list((tmp_path / cfg.HISTORY_DIRNAME).glob("*.json"))
    assert len(backups) == 1
