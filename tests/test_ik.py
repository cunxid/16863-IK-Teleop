"""IK convergence tests on the real robot model (headless, no viewer).

These need the downloaded model; run ``uv run setup-assets`` first.
They are the closest thing to a proof that the math in ik.py is right:
ask for a reachable pose, step the IK, and measure that we actually get there.
"""

import math

import numpy as np
import pytest

from teleop import ik, quat
from teleop.robot import SPECS, Robot
from tests.test_quat import axis_angle

G1 = SPECS["g1"]
pytestmark = pytest.mark.skipif(
    not G1.scene_path().exists(),
    reason="robot model missing — run `uv run setup-assets` first",
)

DT = 1.0 / 60.0


@pytest.fixture()
def robot() -> Robot:
    r = Robot(G1)
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
    frame.left.pos += np.array([0.06, -0.04, 0.05])
    robot.set_targets(frame)

    errors = run_ik(robot, 400)
    assert errors["left"] < 0.005, f"left hand should reach within 5 mm, got {errors['left'] * 1000:.1f} mm"
    # The right arm's target never moved, so the right arm must stay put.
    assert errors["right"] < 1e-4


def test_tracks_orientation(robot: Robot):
    frame = robot.targets()
    frame.left.quat = quat.mul(axis_angle([0, 0, 1], math.radians(30)), frame.left.quat)
    robot.set_targets(frame)

    run_ik(robot, 400)
    err_angle = np.linalg.norm(quat.error_world(robot.target_pose(robot.arms["left"]).quat,
                                                robot.ee_pose(robot.arms["left"]).quat))
    assert err_angle < math.radians(3), f"orientation error {math.degrees(err_angle):.1f}° too large"


def test_unreachable_target_stays_stable(robot: Robot):
    """Ask for the impossible: no NaNs, no limit violations, graceful straining."""
    frame = robot.targets()
    frame.left.pos += np.array([0.8, 0.0, 0.8])  # far outside the arm's reach
    robot.set_targets(frame)

    run_ik(robot, 300)
    q = robot.data.qpos
    assert np.all(np.isfinite(q)), "qpos must stay finite near singularities"
    for arm in robot.arms.values():
        qa = robot.data.qpos[arm.qpos_ids]
        assert np.all(qa >= arm.joint_min - 1e-9) and np.all(qa <= arm.joint_max + 1e-9)


def test_respects_velocity_cap(robot: Robot):
    frame = robot.targets()
    frame.left.pos += np.array([0.3, 0.3, 0.0])
    robot.set_targets(frame)

    settings = ik.IKSettings()
    q_prev = robot.data.qpos.copy()
    ik.step(robot, DT, settings)
    dq = np.abs(robot.data.qpos - q_prev) / DT
    assert np.max(dq) <= settings.max_joint_vel + 1e-9
