import json

import numpy as np
import pytest

from card_scan.homography_read import apply_homography, load_homography


def test_load_homography_raises_if_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_homography(tmp_path / "nope.json")


def test_load_homography_raises_if_h_not_yet_fitted(tmp_path):
    path = tmp_path / "workspace_calib.json"
    path.write_text(json.dumps({"H": None}))
    with pytest.raises(ValueError, match="no fitted homography"):
        load_homography(path)


def test_load_homography_reads_the_matrix(tmp_path):
    path = tmp_path / "workspace_calib.json"
    H = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    path.write_text(json.dumps({"H": H}))
    loaded = load_homography(path)
    assert loaded == pytest.approx(np.array(H))


def test_apply_homography_scales_correctly():
    H = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]])
    assert apply_homography(H, (5.0, 5.0)) == pytest.approx((20.0, 30.0))
