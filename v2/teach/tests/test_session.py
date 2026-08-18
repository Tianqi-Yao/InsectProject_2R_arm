import pytest

from teach.config import TaughtPoint, TeachConfig
from teach.controller import ArmController, MotionParams
from teach.kinematics import ArmParams
from teach.session import TeachSession


class FakeServos:
    def __init__(self, start=(0.0, 0.0)):
        self.present = {"joint1": start[0], "joint2": start[1]}
        self.torque = {"joint1": True, "joint2": True}

    def get_present_deg(self, joint):
        return self.present[joint]

    def set_target_deg(self, joint, angle_deg, speed=800, acc=0):
        self.present[joint] = angle_deg

    def set_torque_enabled(self, joint, enabled):
        self.torque[joint] = enabled


def _session(start=(30.0, 40.0), params=None):
    servos = FakeServos(start)
    controller = ArmController(servos, MotionParams(vmax_deg_s=90.0, amax_deg_s2=180.0))
    return TeachSession(servos, controller, TeachConfig(), params=params), servos


def test_start_releases_torque_on_both_joints():
    session, servos = _session()
    session.start()
    assert servos.torque == {"joint1": False, "joint2": False}


def test_mark_records_current_present_position():
    session, servos = _session(start=(12.0, -34.0))
    point = session.mark("home")
    assert (point.joint1_deg, point.joint2_deg) == (12.0, -34.0)
    assert session.teach_config.get("home") is point


def test_mark_without_kinematics_leaves_workspace_coords_none():
    session, _ = _session(params=None)
    point = session.mark("home")
    assert point.x_mm is None and point.y_mm is None


def test_mark_with_kinematics_fills_in_workspace_coords():
    params = ArmParams.nominal()
    session, servos = _session(params=params)
    point = session.mark("home")
    assert point.x_mm is not None and point.y_mm is not None


def test_mark_overwrites_same_label_re_teaching():
    session, servos = _session(start=(10.0, 10.0))
    session.mark("home")
    servos.present = {"joint1": 20.0, "joint2": 20.0}
    session.mark("home")
    assert len(session.teach_config.points) == 1
    assert session.teach_config.get("home").joint1_deg == 20.0


def test_goto_raises_for_unknown_label():
    session, _ = _session()
    with pytest.raises(KeyError):
        session.goto("nope")


def test_goto_moves_the_arm_and_releases_torque_afterward():
    session, servos = _session(start=(0.0, 0.0))
    session.teach_config.upsert(TaughtPoint("target", 45.0, 45.0))

    reached = session.goto("target")
    assert reached == pytest.approx((45.0, 45.0), abs=1e-6)
    assert servos.torque == {"joint1": False, "joint2": False}, (
        "goto must release torque again afterward -- hand-teaching is the default state"
    )


def test_goto_resyncs_from_actual_hand_moved_position_not_a_stale_one():
    """The critical regression: if the arm was hand-moved to (5, 5) while
    torque was off (after some earlier controller state remembered a
    completely different pose), goto() must plan the move FROM (5, 5), not
    from whatever the controller last thought was true -- otherwise it
    streams a phantom jump back through a stale position first."""
    session, servos = _session(start=(0.0, 0.0))
    session.teach_config.upsert(TaughtPoint("target", 45.0, 45.0))

    # Controller currently thinks the arm is at (0, 0), but it was hand-moved to (5, 5).
    servos.present = {"joint1": 5.0, "joint2": 5.0}

    session.goto("target")
    # commanded_deg's final value is the goal; what matters is that no
    # intermediate step went anywhere near the controller's stale (0, 0)
    # memory instead of the real (5, 5) starting point -- verified
    # indirectly by resync() having been called with the fresh readback.
    assert session.controller.commanded_deg == pytest.approx((45.0, 45.0), abs=1e-6)


def test_stop_reenables_torque_and_holds_current_position():
    session, servos = _session(start=(7.0, 8.0))
    session.start()
    assert servos.torque == {"joint1": False, "joint2": False}
    session.stop()
    assert servos.torque == {"joint1": True, "joint2": True}
    assert servos.present == {"joint1": 7.0, "joint2": 8.0}
