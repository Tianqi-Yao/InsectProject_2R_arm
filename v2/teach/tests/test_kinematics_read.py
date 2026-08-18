import json

from teach.kinematics import ArmParams
from teach.kinematics_read import load_params


def test_load_params_falls_back_to_nominal_when_no_fit_exists(tmp_path):
    params = load_params(tmp_path / "does_not_exist.json")
    assert params == ArmParams.nominal()


def test_load_params_reads_a_fitted_kinematics_calib_file(tmp_path):
    path = tmp_path / "kinematics_calib.json"
    path.write_text(json.dumps({
        "L1": 130.0, "L2": 90.0, "base_x": 95.0, "base_y": -40.0,
        "servo1_offset_deg": 12.0, "servo2_offset_deg": 3.0,
        "servo1_dir": -1, "servo2_dir": 1, "elbow_offset_mm": 7.0,
        "fit_report": {"n_points": 20, "rms_error_mm": 0.5, "max_error_mm": 1.0},
        "ee_tag_id": 10,
    }))
    params = load_params(path)
    assert params.L1 == 130.0
    assert params.servo1_dir == -1
    assert params.elbow_offset_mm == 7.0
