import pytest

from kinematics_fit import config as cfg


def test_default_calib_is_valid_and_elbow_offset_defaults_to_zero():
    calib = cfg.KinematicsCalib()
    cfg.validate(calib)
    assert calib.elbow_offset_mm == 0.0
    assert calib.fit_report is None


def test_round_trip_save_load_including_fit_report(tmp_path):
    path = tmp_path / "kinematics_calib.json"
    calib = cfg.KinematicsCalib(L1=130.0, elbow_offset_mm=12.0,
                                  fit_report=cfg.FitReportSummary(n_points=20, rms_error_mm=0.4,
                                                                    max_error_mm=1.1))
    cfg.save(calib, path)
    loaded = cfg.load(path)
    assert loaded == calib


def test_round_trip_preserves_spotcheck_poses(tmp_path):
    path = tmp_path / "kinematics_calib.json"
    calib = cfg.KinematicsCalib(spotcheck_poses=[{"joint1": 68.0, "joint2": 116.0}])
    cfg.save(calib, path)
    assert cfg.load(path).spotcheck_poses == [{"joint1": 68.0, "joint2": 116.0}]


def test_params_builds_a_consistent_armparams():
    calib = cfg.KinematicsCalib(L1=111.0, L2=88.0, servo1_dir=-1, elbow_offset_mm=5.0)
    p = calib.params()
    assert (p.L1, p.L2, p.servo1_dir, p.elbow_offset_mm) == (111.0, 88.0, -1, 5.0)


def test_validate_rejects_non_positive_link_lengths():
    with pytest.raises(ValueError, match="L1/L2"):
        cfg.validate(cfg.KinematicsCalib(L1=0.0))


def test_validate_rejects_bad_servo_dir():
    with pytest.raises(ValueError, match="servo1_dir"):
        cfg.validate(cfg.KinematicsCalib(servo1_dir=2))


def test_validate_rejects_elbow_offset_out_of_sane_bounds():
    with pytest.raises(ValueError, match="elbow_offset_mm"):
        cfg.validate(cfg.KinematicsCalib(elbow_offset_mm=999.0))


def test_validate_rejects_malformed_spotcheck_pose():
    with pytest.raises(ValueError, match="spotcheck pose"):
        cfg.validate(cfg.KinematicsCalib(spotcheck_poses=[{"joint1": 1.0}]))


def test_load_and_save_honor_a_monkeypatched_default_path(tmp_path, monkeypatch):
    fake_default = tmp_path / "kinematics_calib.json"
    monkeypatch.setattr(cfg, "DEFAULT_PATH", fake_default)
    cfg.save(cfg.KinematicsCalib(L1=142.0))
    assert cfg.load().L1 == 142.0
