"""Unit tests for the scripted lift-in + figure-8 source."""

import numpy as np

from teleop.mapping import Pose
from teleop.sources import TargetFrame
from teleop.sources.trajectory import LIFT_OFFSET, TrajectorySource, smoothstep
from tests.test_quat import IDENTITY

LEFT0 = np.array([0.18, 0.20, 0.45])
RIGHT0 = np.array([0.18, -0.20, 0.45])


def make_source(amplitude=0.1, period=4.0, ramp_time=2.0) -> TrajectorySource:
    initial = TargetFrame(
        left=Pose(LEFT0.copy(), IDENTITY.copy()),
        right=Pose(RIGHT0.copy(), IDENTITY.copy()),
    )
    return TrajectorySource(initial, amplitude=amplitude, period=period, ramp_time=ramp_time)


def test_starts_exactly_at_initial_pose():
    frame = make_source().get_targets(0.0)
    assert np.allclose(frame.left.pos, LEFT0)
    assert np.allclose(frame.right.pos, RIGHT0)


def test_lift_in_reaches_the_working_center():
    frame = make_source(ramp_time=2.0).get_targets(2.0)  # ramp done, u = 0
    assert np.allclose(frame.left.pos, LEFT0 + LIFT_OFFSET)
    assert np.allclose(frame.right.pos, RIGHT0 + LIFT_OFFSET)
    # Orientation is never touched (the K1 tracks position only).
    assert np.allclose(frame.left.quat, IDENTITY)
    assert np.allclose(frame.right.quat, IDENTITY)


def test_smoothstep_shape():
    assert smoothstep(-1.0) == 0.0 and smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0 and smoothstep(2.0) == 1.0
    assert 0.49 < smoothstep(0.5) < 0.51


def test_stays_within_bounds():
    src = make_source(amplitude=0.1, ramp_time=2.0)
    for t in np.linspace(0.0, 10.0, 600):
        frame = src.get_targets(float(t))
        for hand, rest in ((frame.left, LEFT0), (frame.right, RIGHT0)):
            d = hand.pos - rest
            assert -1e-9 <= d[0] <= LIFT_OFFSET[0] + 1e-9  # x: only the lift
            assert abs(d[1]) <= 0.1 + 1e-9  # y: figure-8 width
            assert d[2] <= LIFT_OFFSET[2] + 0.05 + 1e-9  # z: lift + half-amp
            assert d[2] >= -0.05 - 1e-9


def test_motion_is_continuous():
    src = make_source()
    dt = 1.0 / 60.0
    prev = src.get_targets(0.0)
    for i in range(1, 600):
        cur = src.get_targets(i * dt)
        step = np.linalg.norm(cur.left.pos - prev.left.pos)
        assert step < 0.01, "targets should move < 1 cm per 60 Hz frame"
        prev = cur


def test_hands_mirror_each_other():
    frame = make_source().get_targets(3.3)  # mid-figure-8
    left_dy = frame.left.pos[1] - LEFT0[1]
    right_dy = frame.right.pos[1] - RIGHT0[1]
    assert np.isclose(left_dy, -right_dy)
    assert np.isclose(frame.left.pos[2], frame.right.pos[2])  # z in phase
