"""IK convergence tests on the real robot model (headless, no viewer).

These need the downloaded model; run ``uv run setup-assets`` first.
They are the closest thing to a proof that the math in ik.py is right:
ask for a reachable point, step the IK, and measure that we actually get
there. The K1 tracks position only (4-DoF arms — lesson §5's DoF budget),
so orientation assertions check that target rotations are *ignored*.
"""

import math

import numpy as np
import pytest

from teleop import ik
from teleop.robot import SPECS, Robot
from tests.test_quat import axis_angle

K1 = SPECS["k1"]
pytestmark = pytest.mark.skipif(
    not K1.scene_path().exists(),
    reason="robot model missing — run `uv run setup-assets` first",
)

DT = 1.0 / 60.0


@pytest.fixture()
def robot() -> Robot:
    r = Robot(K1)
    r.reset()
    return r


def run_ik(robot: Robot, steps: int) -> dict[str, float]:
    errors: dict[str, float] = {}
    for _ in range(steps):
        errors = ik.step(robot, DT)
    return errors


def test_tracks_at_rest(robot: Robot):
    """Targets start exactly on the hands ⇒ nothing should move."""
    q_before = robot.data.qpos.copy()
    run_ik(robot, 60)
    assert np.allclose(robot.data.qpos, q_before, atol=1e-6)


def test_reaches_nearby_position(robot: Robot):
    frame = robot.targets()
    frame.left.pos += np.array([0.05, -0.03, 0.06])
    robot.set_targets(frame)

    errors = run_ik(robot, 400)
    assert errors["left"] < 0.005, f"left hand should reach within 5 mm, got {errors['left'] * 1000:.1f} mm"
    # The right arm's target never moved, so the right arm must stay put.
    assert errors["right"] < 1e-4


def test_target_rotation_is_ignored(robot: Robot):
    """Position-only arm: spinning the target must not disturb the tracking."""
    frame = robot.targets()
    frame.left.pos += np.array([0.05, 0.0, 0.05])
    frame.left.quat = axis_angle([0, 0, 1], math.radians(90))  # wild rotation
    robot.set_targets(frame)

    errors = run_ik(robot, 400)
    assert errors["left"] < 0.005, "a rotated target must still be reached in position"


def test_unreachable_target_stays_stable(robot: Robot):
    """Ask for the impossible: no NaNs, no limit violations, graceful straining."""
    frame = robot.targets()
    frame.left.pos += np.array([0.5, 0.0, 0.5])  # far outside the arm's ~0.39 m reach
    robot.set_targets(frame)

    run_ik(robot, 300)
    q = robot.data.qpos
    assert np.all(np.isfinite(q)), "qpos must stay finite near singularities"
    for arm in robot.arms.values():
        qa = robot.data.qpos[arm.qpos_ids]
        assert np.all(qa >= arm.joint_min - 1e-9) and np.all(qa <= arm.joint_max + 1e-9)


def test_respects_velocity_cap(robot: Robot):
    frame = robot.targets()
    frame.left.pos += np.array([0.2, 0.2, 0.0])
    robot.set_targets(frame)

    settings = ik.IKSettings()
    q_prev = robot.data.qpos.copy()
    ik.step(robot, DT, settings)
    dq = np.abs(robot.data.qpos - q_prev) / DT
    assert np.max(dq) <= settings.max_joint_vel + 1e-9
