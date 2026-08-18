import pytest

from teach import config as cfg


def test_default_config_is_empty_and_valid():
    calib = cfg.TeachConfig()
    cfg.validate(calib)
    assert calib.points == []


def test_upsert_adds_new_point():
    calib = cfg.TeachConfig()
    calib.upsert(cfg.TaughtPoint("home", 90.0, 90.0))
    assert calib.get("home").joint1_deg == 90.0


def test_upsert_overwrites_existing_label_instead_of_duplicating():
    calib = cfg.TeachConfig()
    calib.upsert(cfg.TaughtPoint("home", 90.0, 90.0))
    calib.upsert(cfg.TaughtPoint("home", 91.0, 92.0))
    assert len(calib.points) == 1
    assert calib.get("home").joint1_deg == 91.0


def test_remove_returns_false_for_unknown_label():
    calib = cfg.TeachConfig()
    assert calib.remove("nope") is False


def test_remove_deletes_point():
    calib = cfg.TeachConfig()
    calib.upsert(cfg.TaughtPoint("home", 90.0, 90.0))
    assert calib.remove("home") is True
    assert calib.get("home") is None


def test_validate_rejects_duplicate_labels():
    calib = cfg.TeachConfig(points=[cfg.TaughtPoint("a", 0, 0), cfg.TaughtPoint("a", 1, 1)])
    with pytest.raises(ValueError, match="duplicate"):
        cfg.validate(calib)


def test_round_trip_save_load(tmp_path):
    path = tmp_path / "teach.json"
    calib = cfg.TeachConfig(points=[cfg.TaughtPoint("home", 90.0, 90.0, x_mm=1.0, y_mm=2.0)])
    cfg.save(calib, path)
    loaded = cfg.load(path)
    assert loaded == calib


def test_save_backs_up_previous_file(tmp_path):
    path = tmp_path / "teach.json"
    cfg.save(cfg.TeachConfig(points=[cfg.TaughtPoint("a", 0, 0)]), path)
    cfg.save(cfg.TeachConfig(points=[cfg.TaughtPoint("b", 1, 1)]), path)
    backups = list((tmp_path / cfg.HISTORY_DIRNAME).glob("*.json"))
    assert len(backups) == 1
