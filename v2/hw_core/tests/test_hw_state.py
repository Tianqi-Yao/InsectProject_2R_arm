import json

import pytest

from arm_hw_core import hw_state as hws


def test_default_state_is_unrestricted_and_valid():
    state = hws.HwState()
    hws.validate(state)  # must not raise
    assert state.joint_limits_deg is None


def test_round_trip_save_load(tmp_path):
    path = tmp_path / "hw_state.json"
    state = hws.HwState(servo_port="/dev/fake", camera_backend="usb", usb_camera_index=2)
    hws.save(state, path)
    loaded = hws.load(path)
    assert loaded == state


def test_load_missing_file_returns_defaults(tmp_path):
    state = hws.load(tmp_path / "does_not_exist.json")
    assert state == hws.HwState()


def test_load_tolerates_unknown_and_missing_keys(tmp_path):
    path = tmp_path / "hw_state.json"
    path.write_text(json.dumps({"servo_port": "/dev/xyz", "some_future_field": 123}))
    state = hws.load(path)
    assert state.servo_port == "/dev/xyz"
    assert state.camera_backend == "usb"  # falls back to the dataclass default


def test_save_backs_up_previous_file_before_overwriting(tmp_path):
    path = tmp_path / "hw_state.json"
    hws.save(hws.HwState(servo_port="/dev/one"), path)
    hws.save(hws.HwState(servo_port="/dev/two"), path)

    history_dir = tmp_path / hws.HISTORY_DIRNAME
    backups = list(history_dir.glob("hw_state_*.json"))
    assert len(backups) == 1
    backed_up = json.loads(backups[0].read_text())
    assert backed_up["servo_port"] == "/dev/one"
    assert hws.load(path).servo_port == "/dev/two"


def test_validate_rejects_unknown_camera_backend():
    state = hws.HwState(camera_backend="not-a-backend")
    with pytest.raises(ValueError, match="camera_backend"):
        hws.validate(state)


def test_validate_rejects_wrong_joint_id_keys():
    state = hws.HwState(joint_ids={"joint1": 1, "joint3": 3})
    with pytest.raises(ValueError, match="joint_ids"):
        hws.validate(state)


def test_validate_rejects_wrapping_joint_range():
    state = hws.HwState(joint_limits_deg={"joint1": [350, 10], "joint2": [0, 360]})
    with pytest.raises(ValueError, match="joint1"):
        hws.validate(state)


def test_validate_rejects_missing_joint_key():
    state = hws.HwState(joint_limits_deg={"joint1": [0, 360]})
    with pytest.raises(ValueError, match="joint2"):
        hws.validate(state)


def test_validate_rejects_undersized_coupled_boundary():
    state = hws.HwState(joint_limits_deg={
        "joint1": [0, 360], "joint2": [0, 360],
        "coupled_boundary": [[1, 1], [2, 2]],
    })
    with pytest.raises(ValueError, match="coupled_boundary"):
        hws.validate(state)


def test_validate_accepts_well_formed_joint_limits():
    state = hws.HwState(joint_limits_deg={
        "joint1": [10, 350], "joint2": [0, 360],
        "coupled_boundary": [[1, 1], [2, 2], [3, 3]],
    })
    hws.validate(state)  # must not raise


def test_save_rejects_invalid_state_without_writing(tmp_path):
    path = tmp_path / "hw_state.json"
    bad = hws.HwState(camera_backend="bogus")
    with pytest.raises(ValueError):
        hws.save(bad, path)
    assert not path.exists()


def test_load_and_save_honor_a_monkeypatched_default_path(tmp_path, monkeypatch):
    """Regression test: load()/save() used to declare `path: Path =
    DEFAULT_PATH` as the parameter default, which binds ONCE at import
    time -- so overriding hw_state.DEFAULT_PATH later (e.g. in a test
    fixture) had no effect on calls made without an explicit path. Both
    functions must resolve DEFAULT_PATH's CURRENT value at call time."""
    fake_default = tmp_path / "hw_state.json"
    monkeypatch.setattr(hws, "DEFAULT_PATH", fake_default)
    hws.save(hws.HwState(servo_port="/dev/patched"))
    assert fake_default.exists()
    assert hws.load().servo_port == "/dev/patched"
