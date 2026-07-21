"""Pure-logic tests for arm_core.py. No hardware needed -- everything here
runs against synthetic data or mocked hw handles."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as ac


# ── 1. Kinematics ─────────────────────────────────────────────────────

def test_ik_matches_firmware_home_point():
    # firmware/2r_arm.ino comment: workspace centre (100,75) -> arm-relative
    # (0, 120mm) -> theta1~=44.4deg, theta2~=115.6deg, s1~=67.5, s2~=115.6
    p = ac.ArmParams.nominal()
    r = ac.ik_solve(p, 100.0, 75.0)
    assert r.reachable
    assert r.theta1_deg == pytest.approx(44.4, abs=0.2)
    assert r.theta2_deg == pytest.approx(115.6, abs=0.2)


def test_ik_matches_firmware_near_right_corner():
    # kinematics.h derivation comment: near-right corner arm-relative (100,45)
    # -> theta1_min ~= -22.98deg (this is what SERVO1_OFFSET=23.08 was tuned against)
    p = ac.ArmParams.nominal()
    r = ac.ik_solve(p, 200.0, 0.0)  # workspace (200,0) -> arm-relative (100,45)
    assert r.reachable
    assert r.theta1_deg == pytest.approx(-22.98, abs=0.2)


def test_ik_fk_round_trip():
    p = ac.ArmParams(L1=118.0, L2=88.0, base_x=97.0, base_y=-42.0,
                      servo1_offset_deg=19.5, servo2_offset_deg=-2.0)
    for x, y in [(100, 75), (30, 20), (170, 130), (60, 110)]:
        r = ac.ik_solve(p, x, y)
        assert r.reachable
        wx, wy = ac.fk_from_servo_angles(p, r.servo1_deg, r.servo2_deg)
        assert wx == pytest.approx(x, abs=1e-6)
        assert wy == pytest.approx(y, abs=1e-6)


def test_ik_fk_round_trip_with_inverted_joint():
    # servo2_dir=-1 models a joint wired/mounted so its raw angle increases
    # opposite to our math convention -- confirmed on real hardware where
    # joint1's jog direction matched expectations but joint2's was reversed.
    p = ac.ArmParams(L1=118.0, L2=88.0, base_x=97.0, base_y=-42.0,
                      servo1_offset_deg=179.3, servo2_offset_deg=179.3,
                      servo1_dir=1, servo2_dir=-1)
    for x, y in [(100, 75), (30, 20), (170, 130), (60, 110)]:
        r = ac.ik_solve(p, x, y)
        assert r.reachable
        wx, wy = ac.fk_from_servo_angles(p, r.servo1_deg, r.servo2_deg)
        assert wx == pytest.approx(x, abs=1e-6)
        assert wy == pytest.approx(y, abs=1e-6)


def test_inverted_joint_direction_actually_changes_servo_command():
    # sanity check that servo2_dir isn't a no-op: flipping it should send a
    # different servo2 command for the same target, not the same one.
    base = dict(L1=118.0, L2=88.0, base_x=97.0, base_y=-42.0,
                servo1_offset_deg=179.3, servo2_offset_deg=179.3)
    p_fwd = ac.ArmParams(**base, servo1_dir=1, servo2_dir=1)
    p_inv = ac.ArmParams(**base, servo1_dir=1, servo2_dir=-1)
    r_fwd = ac.ik_solve(p_fwd, 150.0, 40.0)
    r_inv = ac.ik_solve(p_inv, 150.0, 40.0)
    assert r_fwd.servo1_deg == pytest.approx(r_inv.servo1_deg)
    assert r_fwd.servo2_deg != pytest.approx(r_inv.servo2_deg)


def test_ik_rejects_unreachable_point():
    p = ac.ArmParams.nominal()
    far_beyond_reach = ac.ik_solve(p, 100.0, p.L1 + p.L2 + 500.0)
    assert not far_beyond_reach.reachable


# ── 2. Homography ──────────────────────────────────────────────────────

WORLD_CORNERS = [(0.0, 150.0), (200.0, 150.0), (200.0, 0.0), (0.0, 0.0)]


def _affine_pixel(xy, scale=3.0, offset=(100.0, 50.0)):
    x, y = xy
    return (scale * x + offset[0], scale * y + offset[1])


def test_compute_homography_recovers_clean_affine_mapping():
    pixels = [_affine_pixel(w) for w in WORLD_CORNERS]
    H, rms_px = ac.compute_homography(pixels, WORLD_CORNERS)
    assert rms_px < 1e-6
    for px, world in zip(pixels, WORLD_CORNERS):
        got = ac.apply_homography(H, px)
        assert got[0] == pytest.approx(world[0], abs=1e-4)
        assert got[1] == pytest.approx(world[1], abs=1e-4)


def test_homography_drift_zero_when_nothing_moved():
    pixels = [_affine_pixel(w) for w in WORLD_CORNERS]
    H, _ = ac.compute_homography(pixels, WORLD_CORNERS)
    drift = ac.homography_drift_mm(H, pixels, WORLD_CORNERS)
    assert drift < 1e-6


def test_homography_drift_detects_camera_shift():
    pixels = [_affine_pixel(w) for w in WORLD_CORNERS]
    H, _ = ac.compute_homography(pixels, WORLD_CORNERS)
    # simulate the camera (or the tag sheet) shifting by 30px ~= 10mm at scale=3
    shifted_pixels = [(px + 30.0, py) for px, py in pixels]
    drift = ac.homography_drift_mm(H, shifted_pixels, WORLD_CORNERS)
    assert drift == pytest.approx(10.0, abs=0.5)


# ── 3. Kinematic parameter fit ─────────────────────────────────────────

def test_fit_kinematics_recovers_synthetic_ground_truth():
    true_params = ac.ArmParams(L1=118.0, L2=88.0, base_x=96.0, base_y=-42.0,
                                servo1_offset_deg=19.0, servo2_offset_deg=-3.0)
    rng = np.random.default_rng(42)
    samples = []
    for theta1 in (-10.0, 10.0, 30.0, 50.0):
        for theta2 in (20.0, 60.0, 100.0, 140.0):
            s1 = theta1 + true_params.servo1_offset_deg
            s2 = theta2 + true_params.servo2_offset_deg
            x, y = ac.fk_from_servo_angles(true_params, s1, s2)
            noise = rng.normal(scale=0.2, size=2)  # ~0.2mm measurement noise
            samples.append(ac.CalibSample(s1, s2, x + noise[0], y + noise[1]))

    report = ac.fit_kinematics(samples, x0=ac.ArmParams.nominal())

    assert report.rms_error_mm < 1.0
    fp = report.params
    assert fp.L1 == pytest.approx(true_params.L1, abs=1.5)
    assert fp.L2 == pytest.approx(true_params.L2, abs=1.5)
    assert fp.base_x == pytest.approx(true_params.base_x, abs=1.5)
    assert fp.base_y == pytest.approx(true_params.base_y, abs=1.5)
    assert fp.servo1_offset_deg == pytest.approx(true_params.servo1_offset_deg, abs=1.0)
    assert fp.servo2_offset_deg == pytest.approx(true_params.servo2_offset_deg, abs=1.0)


def test_fit_kinematics_recovers_ground_truth_with_inverted_joint():
    # Same as above but joint2 is wired backwards (servo2_dir=-1), matching
    # the real hardware finding. fit_kinematics must carry dir through from
    # x0 rather than silently assuming +1, or this fit would never converge.
    true_params = ac.ArmParams(L1=118.0, L2=88.0, base_x=96.0, base_y=-42.0,
                                servo1_offset_deg=19.0, servo2_offset_deg=179.3,
                                servo1_dir=1, servo2_dir=-1)
    rng = np.random.default_rng(7)
    samples = []
    for theta1 in (-10.0, 10.0, 30.0, 50.0):
        for theta2 in (20.0, 60.0, 100.0, 140.0):
            s1 = true_params.servo1_offset_deg + true_params.servo1_dir * theta1
            s2 = true_params.servo2_offset_deg + true_params.servo2_dir * theta2
            x, y = ac.fk_from_servo_angles(true_params, s1, s2)
            noise = rng.normal(scale=0.2, size=2)
            samples.append(ac.CalibSample(s1, s2, x + noise[0], y + noise[1]))

    x0 = ac.ArmParams(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                       servo1_offset_deg=23.08, servo2_offset_deg=180.0,
                       servo1_dir=1, servo2_dir=-1)
    report = ac.fit_kinematics(samples, x0=x0)

    assert report.rms_error_mm < 1.0
    fp = report.params
    assert fp.servo2_dir == -1  # carried through, not silently reset to +1
    assert fp.L1 == pytest.approx(true_params.L1, abs=1.5)
    assert fp.L2 == pytest.approx(true_params.L2, abs=1.5)
    assert fp.servo2_offset_deg == pytest.approx(true_params.servo2_offset_deg, abs=1.0)


def test_fit_kinematics_rejects_too_few_samples():
    with pytest.raises(ValueError):
        ac.fit_kinematics([ac.CalibSample(0, 0, 0, 0)] * 3)


def test_generate_calibration_targets_are_reachable_and_shuffled():
    targets = ac.generate_calibration_targets(seed=0)
    assert len(targets) >= 10
    assert all(t.reachable for t in targets)
    assert all(0.0 <= t.servo1_deg <= 180.0 and 0.0 <= t.servo2_deg <= 180.0 for t in targets)


# ── 4. calib.json persistence ──────────────────────────────────────────

def test_save_then_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "DEFAULT_CALIB_PATH", tmp_path / "calib.json")
    monkeypatch.setattr(ac, "CALIB_HISTORY_DIR", tmp_path / "calib_history")
    calib = ac._default_calib()
    ac.save_calib(calib)
    loaded = ac.load_calib()
    assert loaded["kinematics"]["L1"] == calib["kinematics"]["L1"]


def test_save_calib_backs_up_previous_version(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "DEFAULT_CALIB_PATH", tmp_path / "calib.json")
    monkeypatch.setattr(ac, "CALIB_HISTORY_DIR", tmp_path / "calib_history")
    ac.save_calib(ac._default_calib())
    ac.save_calib(ac._default_calib())
    assert list((tmp_path / "calib_history").glob("calib_*.json"))


def test_validate_calib_rejects_out_of_range_param():
    calib = ac._default_calib()
    calib["kinematics"]["L1"] = 9999.0
    with pytest.raises(ValueError):
        ac._validate_calib(calib)


# ── 5. Self-check (mocked hardware) ─────────────────────────────────────

class FakeDetection:
    def __init__(self, center):
        self.center = center


class FakeDetector:
    """Returns a queued sequence of {tag_id: FakeDetection} dicts, one per call."""
    def __init__(self, responses):
        self._responses = list(responses)

    def detect(self, frame):
        return self._responses.pop(0)


class FakeCamera:
    def capture_gray(self):
        return None


class FakeServos:
    def __init__(self, present_deg):
        self._present_deg = present_deg

    def move_and_wait(self, targets_deg):
        pass

    def get_present_deg(self, joint):
        return self._present_deg[joint]


class FakeHw:
    def __init__(self, detector, servos):
        self.camera = FakeCamera()
        self.detector = detector
        self.servos = servos


def _corner_detections(pixel_shift=(0.0, 0.0)):
    pixels = [_affine_pixel(w) for w in WORLD_CORNERS]
    ids = [0, 1, 2, 3]  # tl, tr, br, bl per _default_calib's corner_tag_ids
    return {tid: FakeDetection((px + pixel_shift[0], py + pixel_shift[1]))
            for tid, (px, py) in zip(ids, pixels)}


def _base_calib(tmp_path, monkeypatch, initial_H=True):
    monkeypatch.setattr(ac, "DEFAULT_CALIB_PATH", tmp_path / "calib.json")
    monkeypatch.setattr(ac, "CALIB_HISTORY_DIR", tmp_path / "calib_history")
    monkeypatch.setattr(ac, "ALARMS_LOG_PATH", tmp_path / "alarms.log")
    calib = ac._default_calib()
    if initial_H:
        pixels = [_affine_pixel(w) for w in WORLD_CORNERS]
        H, _ = ac.compute_homography(pixels, WORLD_CORNERS)
        calib["homography"]["H"] = H.tolist()
    return calib


def test_selfcheck_passes_with_no_drift_and_good_spotcheck(tmp_path, monkeypatch):
    calib = _base_calib(tmp_path, monkeypatch)
    params = ac.calib_arm_params(calib)
    pose = calib["spotcheck_poses"][0]
    predicted = ac.fk_from_servo_angles(params, pose["joint1"], pose["joint2"])
    ee_pixel = _affine_pixel(predicted)

    detector = FakeDetector([_corner_detections(), {10: FakeDetection(ee_pixel)}])
    servos = FakeServos({"joint1": pose["joint1"], "joint2": pose["joint2"]})
    hw = FakeHw(detector, servos)

    result = ac.run_selfcheck(hw, calib)
    assert result.ok
    assert result.homography_drift_mm < 0.5
    assert result.spotcheck_errors_mm[0] < 0.5
    assert calib["status"] == "OK"


def test_selfcheck_halts_on_homography_drift(tmp_path, monkeypatch):
    calib = _base_calib(tmp_path, monkeypatch)
    # 30px shift at scale=3px/mm ~= 10mm drift, exceeds default 3mm threshold
    detector = FakeDetector([_corner_detections(pixel_shift=(30.0, 0.0))])
    hw = FakeHw(detector, FakeServos({"joint1": 0, "joint2": 0}))

    result = ac.run_selfcheck(hw, calib)
    assert not result.ok
    assert result.reason == "homography_drift"


def test_selfcheck_halts_on_missing_corner_tag(tmp_path, monkeypatch):
    calib = _base_calib(tmp_path, monkeypatch)
    dets = _corner_detections()
    del dets[1]  # "tr" tag not detected
    detector = FakeDetector([dets])
    hw = FakeHw(detector, FakeServos({"joint1": 0, "joint2": 0}))

    result = ac.run_selfcheck(hw, calib)
    assert not result.ok
    assert "missing_corner_tags" in result.reason


def test_selfcheck_halts_on_arm_position_drift(tmp_path, monkeypatch):
    calib = _base_calib(tmp_path, monkeypatch)
    params = ac.calib_arm_params(calib)
    pose = calib["spotcheck_poses"][0]
    predicted = ac.fk_from_servo_angles(params, pose["joint1"], pose["joint2"])
    # offset the end-effector tag by ~10mm worth of pixels (30px @ scale=3)
    ee_pixel = _affine_pixel((predicted[0] + 10.0, predicted[1]))

    detector = FakeDetector([_corner_detections(), {10: FakeDetection(ee_pixel)}])
    servos = FakeServos({"joint1": pose["joint1"], "joint2": pose["joint2"]})
    hw = FakeHw(detector, servos)

    result = ac.run_selfcheck(hw, calib)
    assert not result.ok
    assert result.reason == "arm_position_drift"


def test_selfcheck_first_boot_has_no_prior_homography_to_compare(tmp_path, monkeypatch):
    calib = _base_calib(tmp_path, monkeypatch, initial_H=False)
    params = ac.calib_arm_params(calib)
    pose = calib["spotcheck_poses"][0]
    predicted = ac.fk_from_servo_angles(params, pose["joint1"], pose["joint2"])
    ee_pixel = _affine_pixel(predicted)

    detector = FakeDetector([_corner_detections(), {10: FakeDetection(ee_pixel)}])
    servos = FakeServos({"joint1": pose["joint1"], "joint2": pose["joint2"]})
    hw = FakeHw(detector, servos)

    result = ac.run_selfcheck(hw, calib)
    assert result.ok
    assert result.homography_drift_mm is None
