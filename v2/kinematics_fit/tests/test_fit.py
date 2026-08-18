import numpy as np
import pytest

from kinematics_fit.angles import normalize_deg
from kinematics_fit.fit import CalibSample, fit_kinematics, generate_calibration_targets
from kinematics_fit.kinematics import ArmParams, fk_from_servo_angles


def _synthetic_samples(true_params: ArmParams, n=20, seed=0) -> list[CalibSample]:
    rng = np.random.default_rng(seed)
    samples = []
    for s1, s2 in zip(rng.uniform(0, 360, n), rng.uniform(0, 360, n)):
        x, y = fk_from_servo_angles(true_params, s1, s2)
        samples.append(CalibSample(s1, s2, x, y))
    return samples


def test_fit_kinematics_requires_at_least_6_samples():
    with pytest.raises(ValueError, match=">=6"):
        fit_kinematics([CalibSample(0, 0, 0, 0)] * 5)


def test_fit_kinematics_recovers_ground_truth_from_noise_free_data():
    true_params = ArmParams(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                              servo1_offset_deg=23.08, servo2_offset_deg=5.0)
    samples = _synthetic_samples(true_params, n=25)
    report = fit_kinematics(samples, x0=ArmParams.nominal())

    assert report.rms_error_mm == pytest.approx(0.0, abs=1e-3)
    p = report.params
    assert p.L1 == pytest.approx(true_params.L1, abs=1e-2)
    assert p.L2 == pytest.approx(true_params.L2, abs=1e-2)
    assert p.base_x == pytest.approx(true_params.base_x, abs=1e-2)
    assert p.base_y == pytest.approx(true_params.base_y, abs=1e-2)


def test_fit_kinematics_cannot_separate_elbow_offset_from_l1_but_fit_stays_correct():
    """The key regression guardrail for the documented non-identifiability
    (see kinematics.ArmParams' docstring): elbow_offset_mm is NOT part of
    the fit vector, so generating synthetic data with a nonzero
    elbow_offset_mm and fitting with the standard 6-parameter vector
    (using the SAME true elbow_offset_mm as a fixed input, exactly like a
    real hand-measured constant would be supplied) must still converge to
    near-zero residual error. This proves the degeneracy is harmless
    *because* elbow_offset_mm is excluded from the fit -- it is NOT a test
    that the fit can recover elbow_offset_mm itself (it deliberately
    cannot, and must never be asked to: adding it to _PARAM_ORDER would
    silently make the fit ill-posed, trading L1 against it arbitrarily)."""
    true_params = ArmParams(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                              servo1_offset_deg=23.08, servo2_offset_deg=5.0,
                              elbow_offset_mm=28.0)
    samples = _synthetic_samples(true_params, n=25)

    # x0 supplies the SAME elbow_offset_mm as a fixed, hand-measured input
    # -- exactly how a real caller would use this (see config.KinematicsCalib).
    x0 = ArmParams.nominal()
    x0.elbow_offset_mm = 28.0
    report = fit_kinematics(samples, x0=x0)

    assert report.rms_error_mm == pytest.approx(0.0, abs=1e-2)
    assert report.params.elbow_offset_mm == 28.0  # carried through untouched, never fit


def test_fit_kinematics_with_wrong_fixed_elbow_offset_still_fits_but_shifts_l1():
    """If the WRONG elbow_offset_mm is supplied (e.g. a caliper measurement
    error), the fit still converges to near-zero error by trading it
    against L1/servo offsets -- this is the degeneracy itself, demonstrated
    directly: it's why elbow_offset_mm must be measured carefully and
    trusted, not inferred from fit quality."""
    true_params = ArmParams(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                              servo1_offset_deg=23.08, servo2_offset_deg=5.0,
                              elbow_offset_mm=28.0)
    samples = _synthetic_samples(true_params, n=25)

    x0 = ArmParams.nominal()
    x0.elbow_offset_mm = 4.0  # deliberately wrong
    report = fit_kinematics(samples, x0=x0)

    assert report.rms_error_mm == pytest.approx(0.0, abs=1e-2)  # still fits!
    assert report.params.L1 != pytest.approx(true_params.L1, abs=1.0)  # but L1 is wrong


def test_generate_calibration_targets_are_all_reachable_and_shuffled():
    params = ArmParams.nominal()
    targets = generate_calibration_targets(params=params, nx=6, ny=5, seed=42)
    assert len(targets) > 0
    assert all(t.reachable for t in targets)

    targets_seed_a = generate_calibration_targets(params=params, nx=6, ny=5, seed=1)
    targets_seed_b = generate_calibration_targets(params=params, nx=6, ny=5, seed=2)
    order_a = [(t.servo1_deg, t.servo2_deg) for t in targets_seed_a]
    order_b = [(t.servo1_deg, t.servo2_deg) for t in targets_seed_b]
    assert order_a != order_b  # different seeds shuffle differently
    assert sorted(order_a) == sorted(order_b)  # but cover the same set of points


def test_generate_calibration_targets_respects_joint_limits():
    params = ArmParams.nominal()
    unrestricted = generate_calibration_targets(params=params, nx=6, ny=5, seed=0)
    limits = {"joint1": (0.0, 90.0), "joint2": (0.0, 360.0)}
    restricted = generate_calibration_targets(params=params, nx=6, ny=5, seed=0, joint_limits=limits)
    assert len(restricted) < len(unrestricted)
    assert all(0.0 <= normalize_deg(t.servo1_deg) <= 90.0 for t in restricted)
