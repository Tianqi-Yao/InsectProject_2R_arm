from arm_hw_core.limits import normalize_deg, point_in_polygon, within_joint_limits

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_point_in_polygon_square():
    assert point_in_polygon(5.0, 5.0, SQUARE) is True
    assert point_in_polygon(-1.0, 5.0, SQUARE) is False
    assert point_in_polygon(15.0, 5.0, SQUARE) is False


def test_point_in_polygon_triangle():
    triangle = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
    assert point_in_polygon(5.0, 3.0, triangle) is True
    assert point_in_polygon(9.0, 9.0, triangle) is False


def test_point_in_polygon_concave_shape():
    # A "C" / concave notch cut out of a square on the right side.
    concave = [(0, 0), (10, 0), (10, 4), (5, 4), (5, 6), (10, 6), (10, 10), (0, 10)]
    assert point_in_polygon(2.0, 5.0, concave) is True   # inside the body
    assert point_in_polygon(8.0, 5.0, concave) is False  # inside the notch


def test_point_in_polygon_retraced_loop_uses_winding_number_not_even_odd():
    """Regression test for the corrected mistake documented in limits.py:
    a hand-traced boundary can double back over the same edge. Tracing a
    simple square's perimeter TWICE (an out-and-back retrace of one edge)
    must still report interior points as inside -- an even-odd
    implementation would flip them to "outside" purely because of the lap
    parity, which is exactly the bug this winding-number implementation
    was written to avoid."""
    # Walk the square once, then re-walk the same edge back to a vertex and
    # continue around again -- simulating a hand that hesitated/backtracked
    # mid-trace, baked into the recorded vertex list verbatim.
    retraced = [
        (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0),
        (10.0, 0.0), (10.0, 10.0), (0.0, 10.0),
    ]
    assert point_in_polygon(5.0, 5.0, retraced) is True
    assert point_in_polygon(-1.0, 5.0, retraced) is False


def test_normalize_deg_wraps_to_0_360():
    assert normalize_deg(-10.0) == 350.0
    assert normalize_deg(370.0) == 10.0
    assert normalize_deg(0.0) == 0.0


def test_within_joint_limits_none_is_unrestricted():
    assert within_joint_limits(1000.0, -1000.0, None) is True


def test_within_joint_limits_independent_ranges():
    limits = {"joint1": (10.0, 100.0), "joint2": (0.0, 50.0)}
    assert within_joint_limits(50.0, 25.0, limits) is True
    assert within_joint_limits(5.0, 25.0, limits) is False   # joint1 out of range
    assert within_joint_limits(50.0, 60.0, limits) is False  # joint2 out of range


def test_within_joint_limits_normalizes_before_checking():
    limits = {"joint1": (10.0, 100.0), "joint2": (0.0, 50.0)}
    assert within_joint_limits(50.0 + 360.0, 25.0 - 360.0, limits) is True


def test_within_joint_limits_empty_coupled_boundary_is_a_no_op():
    limits = {"joint1": (0.0, 360.0), "joint2": (0.0, 360.0), "coupled_boundary": []}
    assert within_joint_limits(50.0, 50.0, limits) is True


def test_within_joint_limits_coupled_boundary_rejects_outside_points():
    limits = {
        "joint1": (0.0, 360.0),
        "joint2": (0.0, 360.0),
        "coupled_boundary": SQUARE,
    }
    assert within_joint_limits(5.0, 5.0, limits) is True
    assert within_joint_limits(50.0, 50.0, limits) is False
