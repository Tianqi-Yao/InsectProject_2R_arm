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


def test_fk_with_elbow_offset_matches_manual_geometry():
    # theta1=0, elbow_offset_mm=10, L1=100: joint2's real axis is the third
    # corner of a 10-100-hypot(10,100) right triangle off joint1's axis, at
    # angle atan2(10,100) above the L1=0 direction -- verify fk_joint_positions
    # actually places it there rather than at (L1, 0).
    p = ac.ArmParams(L1=100.0, L2=50.0, base_x=0.0, base_y=0.0,
                      servo1_offset_deg=0.0, servo2_offset_deg=0.0,
                      elbow_offset_mm=10.0)
    elbow, _ee = ac.fk_joint_positions(p, servo1_deg=0.0, servo2_deg=0.0)
    reach = math.hypot(100.0, 10.0)
    angle = math.atan2(10.0, 100.0)
    assert elbow[0] == pytest.approx(reach * math.cos(angle), abs=1e-9)
    assert elbow[1] == pytest.approx(reach * math.sin(angle), abs=1e-9)
    # elbow_offset_mm=0 would have placed it at exactly (100, 0) -- confirm
    # the offset actually moved it, not a no-op.
    assert elbow[1] != pytest.approx(0.0, abs=1e-6)


def test_ik_fk_round_trip_with_elbow_offset():
    p = ac.ArmParams(L1=118.0, L2=88.0, base_x=97.0, base_y=-42.0,
                      servo1_offset_deg=19.5, servo2_offset_deg=-2.0,
                      elbow_offset_mm=28.0)
    for x, y in [(100, 75), (30, 20), (170, 130), (60, 110)]:
        r = ac.ik_solve(p, x, y)
        assert r.reachable
        wx, wy = ac.fk_from_servo_angles(p, r.servo1_deg, r.servo2_deg)
        assert wx == pytest.approx(x, abs=1e-6)
        assert wy == pytest.approx(y, abs=1e-6)


def test_ik_fk_round_trip_with_negative_elbow_offset():
    # the offset's sign matters (which side of L1's line joint2's axis sits
    # on) -- round-trip should hold regardless of which side.
    p = ac.ArmParams(L1=118.0, L2=88.0, base_x=97.0, base_y=-42.0,
                      servo1_offset_deg=19.5, servo2_offset_deg=-2.0,
                      elbow_offset_mm=-28.0)
    for x, y in [(100, 75), (30, 20), (170, 130), (60, 110)]:
        r = ac.ik_solve(p, x, y)
        assert r.reachable
        wx, wy = ac.fk_from_servo_angles(p, r.servo1_deg, r.servo2_deg)
        assert wx == pytest.approx(x, abs=1e-6)
        assert wy == pytest.approx(y, abs=1e-6)


def test_elbow_offset_zero_is_colinear():
    # elbow_offset_mm=0.0 (the default) must reduce to the plain colinear
    # formula -- this is the backward-compatibility case for every existing
    # arm/calib.json that predates this field.
    p0 = ac.ArmParams(L1=118.0, L2=88.0, base_x=97.0, base_y=-42.0,
                       servo1_offset_deg=19.5, servo2_offset_deg=-2.0)
    elbow, ee = ac.fk_joint_positions(p0, servo1_deg=40.0, servo2_deg=90.0)
    theta1 = math.radians(40.0 - 19.5)
    theta2 = math.radians(90.0 - (-2.0))
    expected_elbow = (97.0 + 118.0 * math.cos(theta1), -42.0 + 118.0 * math.sin(theta1))
    assert elbow[0] == pytest.approx(expected_elbow[0], abs=1e-9)
    assert elbow[1] == pytest.approx(expected_elbow[1], abs=1e-9)


def test_ik_rejects_unreachable_point():
    p = ac.ArmParams.nominal()
    far_beyond_reach = ac.ik_solve(p, 100.0, p.L1 + p.L2 + 500.0)
    assert not far_beyond_reach.reachable


# ── Joint limits (mechanical dead-zone protection) ──────────────────

def test_within_joint_limits_none_means_unrestricted():
    assert ac.within_joint_limits(9999.0, -9999.0, None)


def test_within_joint_limits_accepts_inside_range():
    limits = {"joint1": (10.0, 170.0), "joint2": (20.0, 340.0)}
    assert ac.within_joint_limits(90.0, 180.0, limits)


def test_within_joint_limits_rejects_outside_range():
    limits = {"joint1": (10.0, 170.0), "joint2": (20.0, 340.0)}
    assert not ac.within_joint_limits(5.0, 180.0, limits)  # joint1 too low
    assert not ac.within_joint_limits(90.0, 350.0, limits)  # joint2 too high


def test_within_joint_limits_normalizes_out_of_0_360_input():
    # -20.5deg and 339.5deg are the same physical angle; a limit of
    # [300, 360] should accept the raw (unwrapped) -20.5 value too, since
    # that's what set_target_deg's own `% TICKS_PER_REV` would resolve it to.
    limits = {"joint1": (300.0, 360.0), "joint2": (0.0, 360.0)}
    assert ac.within_joint_limits(-20.5, 0.0, limits)


def test_ik_solve_rejects_target_outside_joint_limits():
    p = ac.ArmParams.nominal()
    r_unrestricted = ac.ik_solve(p, 100.0, 75.0)
    assert r_unrestricted.reachable
    # construct a limit that specifically excludes the servo1 angle we just got
    tight_limits = {"joint1": (r_unrestricted.servo1_deg + 5.0, 360.0), "joint2": (0.0, 360.0)}
    r_restricted = ac.ik_solve(p, 100.0, 75.0, joint_limits=tight_limits)
    assert not r_restricted.reachable


def test_generate_calibration_targets_respects_joint_limits():
    p = ac.ArmParams.nominal()
    unrestricted = ac.generate_calibration_targets(params=p, seed=0)
    assert len(unrestricted) > 0
    # a limit tight enough that nothing should pass
    impossible_limits = {"joint1": (0.0, 0.001), "joint2": (0.0, 0.001)}
    restricted = ac.generate_calibration_targets(params=p, seed=0, joint_limits=impossible_limits)
    assert restricted == []


def test_calib_joint_limits_none_by_default():
    calib = ac._default_calib()
    assert ac.calib_joint_limits(calib) is None


def test_calib_joint_limits_reads_configured_values():
    calib = ac._default_calib()
    calib["joint_limits_deg"] = {"joint1": [10.0, 170.0], "joint2": [20.0, 340.0]}
    limits = ac.calib_joint_limits(calib)
    assert limits == {"joint1": (10.0, 170.0), "joint2": (20.0, 340.0), "coupled_boundary": []}


def test_calib_joint_limits_reads_coupled_boundary_in_traced_order():
    calib = ac._default_calib()
    calib["joint_limits_deg"] = {
        "joint1": [0.0, 180.0], "joint2": [20.0, 340.0],
        "coupled_boundary": [
            {"joint1": 90.0, "joint2": 150.0},
            {"joint1": 20.0, "joint2": 100.0},
            {"joint1": 20.0, "joint2": 340.0},
        ],
    }
    limits = ac.calib_joint_limits(calib)
    # order is preserved exactly as traced -- point-in-polygon doesn't care
    # about sort order, unlike the old interpolated-interval model did.
    assert limits["coupled_boundary"] == [(90.0, 150.0), (20.0, 100.0), (20.0, 340.0)]


# ── Coupled/relative dead zones (distal link's range depends on proximal) ──
#
# This models a *continuously* varying relationship (joint2's clearance
# changes smoothly as joint1 sweeps), not a discrete on/off zone -- a
# closed polygon traced by hand around the whole safe region's perimeter,
# with membership decided by point-in-polygon (inside = safe). See
# manual_test/trace_boundary_gui.py / main.py's set-joint-limits command.

def test_point_in_polygon_square():
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert ac._point_in_polygon(5.0, 5.0, square)
    assert not ac._point_in_polygon(15.0, 5.0, square)
    assert not ac._point_in_polygon(-1.0, 5.0, square)
    assert not ac._point_in_polygon(5.0, 15.0, square)


def test_point_in_polygon_triangle():
    triangle = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
    assert ac._point_in_polygon(5.0, 3.0, triangle)
    assert not ac._point_in_polygon(1.0, 9.0, triangle)
    assert not ac._point_in_polygon(9.0, 9.0, triangle)


def test_point_in_polygon_concave_shape():
    # a "C"/notch shape -- a simple min/max-per-slice model can't represent
    # this at all, which is exactly why the polygon check replaced it.
    notch = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (6.0, 10.0),
             (6.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0)]
    assert ac._point_in_polygon(2.0, 8.0, notch)   # left arm of the C
    assert ac._point_in_polygon(8.0, 8.0, notch)   # right arm of the C
    assert not ac._point_in_polygon(5.0, 8.0, notch)  # inside the notch itself


def test_point_in_polygon_robust_to_retracing_the_same_loop():
    # A hand-traced boundary is saved verbatim, hesitations/backtracks and
    # all -- if the recording happens to retrace the whole loop more than
    # once (e.g. a moment of doubling back over already-walked ground),
    # the even-odd/ray-casting rule would flip its verdict depending on
    # whether the lap count is even or odd (provably wrong: an EVEN number
    # of full retraces reads as "outside" for every point the loop
    # encloses, purely by coincidence of parity). Winding number doesn't
    # have this failure mode: 1 lap or 5 laps around the same point both
    # read as "inside."
    square = [(100.0, 200.0), (100.0, 150.0), (200.0, 150.0), (200.0, 200.0)]
    inside_pt, outside_pt = (150.0, 175.0), (50.0, 175.0)
    for reps in (1, 2, 3, 4, 5):
        traced = square * reps
        assert ac._point_in_polygon(*inside_pt, traced), f"reps={reps}"
        assert not ac._point_in_polygon(*outside_pt, traced), f"reps={reps}"


def test_within_joint_limits_coupled_boundary_inside_and_outside():
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 200.0), (0.0, 200.0)]
    limits = {"joint1": (0.0, 180.0), "joint2": (0.0, 360.0), "coupled_boundary": square}
    assert ac.within_joint_limits(50.0, 100.0, limits)
    assert not ac.within_joint_limits(150.0, 100.0, limits)  # outside the traced polygon


def test_within_joint_limits_coupled_boundary_combines_with_independent_ranges():
    # the polygon alone would allow this point, but joint2's own
    # unconditional range is tighter -- AND semantics, tighter one wins.
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 360.0), (0.0, 360.0)]
    limits = {"joint1": (0.0, 180.0), "joint2": (0.0, 170.0), "coupled_boundary": square}
    assert not ac.within_joint_limits(50.0, 200.0, limits)
    assert ac.within_joint_limits(50.0, 100.0, limits)


def test_within_joint_limits_empty_coupled_boundary_is_a_no_op():
    limits = {"joint1": (0.0, 180.0), "joint2": (0.0, 360.0), "coupled_boundary": []}
    assert ac.within_joint_limits(90.0, 180.0, limits)


def test_validate_calib_accepts_valid_coupled_boundary():
    calib = ac._default_calib()
    calib["joint_limits_deg"] = {
        "joint1": [0.0, 360.0], "joint2": [0.0, 360.0],
        "coupled_boundary": [{"joint1": 20.0, "joint2": 10.0},
                             {"joint1": 100.0, "joint2": 10.0},
                             {"joint1": 60.0, "joint2": 200.0}],
    }
    ac._validate_calib(calib)  # should not raise


def test_validate_calib_rejects_boundary_with_fewer_than_3_vertices():
    calib = ac._default_calib()
    calib["joint_limits_deg"] = {
        "joint1": [0.0, 360.0], "joint2": [0.0, 360.0],
        "coupled_boundary": [{"joint1": 20.0, "joint2": 10.0},
                             {"joint1": 100.0, "joint2": 10.0}],  # only 2 -- not a polygon
    }
    with pytest.raises(ValueError):
        ac._validate_calib(calib)


def test_validate_calib_rejects_boundary_vertex_missing_joint2():
    calib = ac._default_calib()
    calib["joint_limits_deg"] = {
        "joint1": [0.0, 360.0], "joint2": [0.0, 360.0],
        "coupled_boundary": [{"joint1": 20.0, "joint2": 10.0},
                             {"joint1": 100.0},
                             {"joint1": 60.0, "joint2": 200.0}],
    }
    with pytest.raises(ValueError):
        ac._validate_calib(calib)


def test_validate_calib_accepts_boundary_vertex_wider_than_absolute_ranges():
    # not a data-entry mistake: the tool that captures this polygon widens
    # joint1/joint2's own independent ranges to the union of themselves and
    # every traced vertex before saving (hand sweeps aren't perfectly
    # repeatable between separate passes) -- so a vertex sitting right at
    # the edge of (or a hair past) the independent range is expected, not
    # rejected.
    calib = ac._default_calib()
    calib["spotcheck_poses"] = []
    calib["joint_limits_deg"] = {
        "joint1": [10.0, 170.0], "joint2": [10.0, 170.0],
        "coupled_boundary": [{"joint1": 5.0, "joint2": 5.0},  # outside both ranges above
                             {"joint1": 175.0, "joint2": 5.0},
                             {"joint1": 90.0, "joint2": 175.0}],
    }
    ac._validate_calib(calib)  # should not raise


def test_validate_calib_rejects_wrapping_joint_limits():
    calib = ac._default_calib()
    calib["joint_limits_deg"] = {"joint1": [350.0, 10.0], "joint2": [0.0, 360.0]}
    with pytest.raises(ValueError):
        ac._validate_calib(calib)


def test_validate_calib_rejects_spotcheck_pose_outside_joint_limits():
    calib = ac._default_calib()
    pose = calib["spotcheck_poses"][0]
    # a limit that specifically excludes the default spotcheck pose
    calib["joint_limits_deg"] = {"joint1": [pose["joint1"] + 5.0, 360.0], "joint2": [0.0, 360.0]}
    with pytest.raises(ValueError):
        ac._validate_calib(calib)


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


def test_fit_kinematics_cannot_separate_elbow_offset_from_l1():
    # Documents an exact degeneracy (not just poor conditioning): vision
    # data can only ever pin down reach=hypot(L1, elbow_offset_mm), never
    # the individual split -- trading L1 against elbow_offset_mm is exactly
    # absorbable by servo1_offset_deg/servo2_offset_deg (see ArmParams'
    # docstring). So elbow_offset_mm is carried through fit_kinematics as a
    # FIXED value from x0, never touched by the optimizer -- confirm that's
    # true, rather than silently drifting to some other value that happens
    # to also fit well.
    true_params = ac.ArmParams(L1=118.0, L2=88.0, base_x=96.0, base_y=-42.0,
                                servo1_offset_deg=19.0, servo2_offset_deg=-3.0,
                                elbow_offset_mm=28.0)
    rng = np.random.default_rng(7)
    samples = []
    for theta1 in (-40.0, -10.0, 10.0, 30.0, 50.0, 80.0):
        for theta2 in (20.0, 60.0, 100.0, 140.0):
            s1 = theta1 + true_params.servo1_offset_deg
            s2 = theta2 + true_params.servo2_offset_deg
            x, y = ac.fk_from_servo_angles(true_params, s1, s2)
            noise = rng.normal(scale=0.2, size=2)
            samples.append(ac.CalibSample(s1, s2, x + noise[0], y + noise[1]))

    x0 = ac.ArmParams.nominal()
    x0.elbow_offset_mm = 28.0
    report = ac.fit_kinematics(samples, x0=x0)

    assert report.rms_error_mm < 1.0
    assert report.params.elbow_offset_mm == 28.0  # untouched, carried straight through


# ── Quick fold-to-known-angle servo2_offset_deg fix ─────────────────────

def test_servo2_offset_from_known_elbow_angle_recovers_true_offset():
    # fold the arm so the elbow reads 90deg (protractor convention: 180=straight)
    # -> theta2=90 -- verify the encoder reading at that exact pose recovers
    # the true servo2_offset_deg used to compute it.
    true_offset = -7.5
    servo2_dir = 1
    theta2_at_fold = 90.0
    servo2_deg = true_offset + servo2_dir * theta2_at_fold
    recovered = ac.servo2_offset_from_known_elbow_angle(servo2_deg, servo2_dir, elbow_angle_deg=90.0)
    assert recovered == pytest.approx(true_offset)


def test_servo2_offset_from_known_elbow_angle_with_inverted_dir():
    true_offset = 145.0
    servo2_dir = -1
    theta2_at_fold = 60.0  # elbow_angle_deg=120 -> theta2_target=60
    servo2_deg = true_offset + servo2_dir * theta2_at_fold
    recovered = ac.servo2_offset_from_known_elbow_angle(servo2_deg, servo2_dir, elbow_angle_deg=120.0)
    assert recovered == pytest.approx(true_offset)


def test_servo2_offset_from_known_elbow_angle_flip_recovers_opposite_fold():
    # a bare protractor reading can't tell which rotational way the elbow
    # was folded -- if the true fold was theta2=-90 (not the assumed +90),
    # flip=True should recover the correct offset instead.
    true_offset = 10.0
    servo2_dir = 1
    theta2_at_fold = -90.0  # folded the "other way"
    servo2_deg = true_offset + servo2_dir * theta2_at_fold

    wrong = ac.servo2_offset_from_known_elbow_angle(servo2_deg, servo2_dir, elbow_angle_deg=90.0, flip=False)
    assert wrong != pytest.approx(true_offset)

    right = ac.servo2_offset_from_known_elbow_angle(servo2_deg, servo2_dir, elbow_angle_deg=90.0, flip=True)
    assert right == pytest.approx(true_offset)


def test_servo2_offset_from_known_elbow_angle_straight_pose():
    # elbow_angle_deg=180 (fully straight) -> theta2_target=0, so
    # servo2_offset_deg is just whatever the raw reading was at that pose.
    servo2_dir = 1
    servo2_deg = 42.0
    recovered = ac.servo2_offset_from_known_elbow_angle(servo2_deg, servo2_dir, elbow_angle_deg=180.0)
    assert recovered == pytest.approx(42.0)


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


def test_generate_scan_path_starts_top_left_and_snakes():
    path = ac.generate_scan_path(width_mm=200.0, height_mm=150.0, nx=5, ny=4, margin_mm=20.0)
    assert len(path) == 20
    # top-left: smallest x, largest y (matches corner_world_mm's "tl" = [0,150] convention)
    x0, y0, _ = path[0]
    assert x0 == pytest.approx(20.0)
    assert y0 == pytest.approx(130.0)
    # row 1 (indices 0-4) ascends in x, row 2 (indices 5-9) descends -- serpentine
    row1_xs = [p[0] for p in path[0:5]]
    row2_xs = [p[0] for p in path[5:10]]
    assert row1_xs == sorted(row1_xs)
    assert row2_xs == sorted(row2_xs, reverse=True)


def test_generate_scan_path_rows_limit_truncates_without_changing_spacing():
    full = ac.generate_scan_path(nx=5, ny=4, margin_mm=20.0)
    limited = ac.generate_scan_path(nx=5, ny=4, margin_mm=20.0, rows_limit=2)
    assert len(limited) == 10
    # same first 10 points, same spacing -- just stops early
    assert limited == full[:10]


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


def test_calib_hardware_config_defaults_match_confirmed_joint_ids():
    # joint2's real bus ID was confirmed as 2 (not the stale 4 that used to
    # live in software/main.py) -- this default is now the single source
    # of truth all frontends read from.
    hw_cfg = ac.calib_hardware_config(ac._default_calib())
    assert hw_cfg.joint_ids == {"joint1": 1, "joint2": 2}


def test_calib_hardware_config_backward_compat_missing_section():
    # An old calib.json predating the "hardware" section shouldn't blow up.
    calib = ac._default_calib()
    del calib["hardware"]
    hw_cfg = ac.calib_hardware_config(calib)
    assert hw_cfg.servo_port == "/dev/cu.usbserial-0001"
    assert hw_cfg.joint_ids == {"joint1": 1, "joint2": 2}


def test_calib_motion_config_backward_compat_missing_section():
    calib = ac._default_calib()
    del calib["motion"]
    motion_cfg = ac.calib_motion_config(calib)
    assert motion_cfg.planner_name == "trapezoidal"
    assert motion_cfg.scan_nx == 50


def test_calib_motion_config_overrides_merge_with_defaults():
    calib = ac._default_calib()
    calib["motion"]["scan_nx"] = 10  # partial override, e.g. from a hand-edited calib.json
    motion_cfg = ac.calib_motion_config(calib)
    assert motion_cfg.scan_nx == 10
    assert motion_cfg.scan_ny == 40  # untouched fields keep their defaults


def test_validate_calib_does_not_require_hardware_or_motion_sections():
    calib = ac._default_calib()
    del calib["hardware"]
    del calib["motion"]
    ac._validate_calib(calib)  # should not raise


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
