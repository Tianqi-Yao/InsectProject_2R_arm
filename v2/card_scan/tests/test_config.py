import pytest

from card_scan import config as cfg


def test_default_config_is_valid():
    calib = cfg.CardScanConfig()
    cfg.validate(calib)
    assert calib.tag_id is None


def test_validate_rejects_rows_or_cols_below_2():
    with pytest.raises(ValueError, match=">=2"):
        cfg.validate(cfg.CardScanConfig(rows=1))


def test_validate_rejects_partially_configured_tag():
    with pytest.raises(ValueError, match="tag_id"):
        cfg.validate(cfg.CardScanConfig(tag_id=20, width_mm=None, height_mm=54.0))


def test_validate_accepts_fully_configured_tag():
    cfg.validate(cfg.CardScanConfig(tag_id=20, width_mm=85.6, height_mm=54.0))


def test_validate_rejects_one_sided_manual_corner():
    with pytest.raises(ValueError, match="manual_corner"):
        cfg.validate(cfg.CardScanConfig(manual_corner_a_mm=[0.0, 0.0]))


def test_round_trip_save_load(tmp_path):
    path = tmp_path / "card_scan.json"
    calib = cfg.CardScanConfig(tag_id=20, width_mm=85.6, height_mm=54.0, rows=4, cols=5,
                                 manual_corner_a_mm=[0.0, 0.0], manual_corner_b_mm=[85.6, 54.0])
    cfg.save(calib, path)
    assert cfg.load(path) == calib


def test_save_backs_up_previous_file(tmp_path):
    path = tmp_path / "card_scan.json"
    cfg.save(cfg.CardScanConfig(rows=3), path)
    cfg.save(cfg.CardScanConfig(rows=4), path)
    backups = list((tmp_path / cfg.HISTORY_DIRNAME).glob("*.json"))
    assert len(backups) == 1
