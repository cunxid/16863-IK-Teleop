"""Differential inverse kinematics — the "brain" of the whole project.

The question IK answers every frame (60×/second):

    "The hand is HERE, the target is THERE.
     How should each arm joint turn, a tiny bit, to close the gap?"

We never solve for a full goal configuration. We only compute a joint
*velocity* that moves the end-effector toward the target, take one small
step, and repeat. Tracking a moving VR controller then comes for free.

The three ideas, in the order they appear below:

1. **Pose error → desired twist.** Position error is a subtraction;
   orientation error is a rotation vector (see ``quat.error_world``).
   Multiply by gains: "close this fraction of the error per second".
2. **Jacobian.** MuJoCo hands us J, the matrix mapping joint velocities to
   end-effector velocity. We keep only this arm's columns.
3. **Damped least squares (DLS).** Instead of inverting J (explodes near
   singularities, e.g. a straight elbow), solve a *damped* version:
   dq = Jᵀ (J Jᵀ + λ²I)⁻¹ · twist. The damping λ trades a little accuracy
   for a lot of stability.

Plus one optional refinement: any leftover freedom is spent drifting toward a
comfortable home posture — in the *nullspace*, so it never disturbs the hand.

**The DoF budget.** A full pose is 6 numbers (3 position + 3 orientation), so
tracking both needs at least 6 joints per arm. The K1 has 4 (and one of them
is forearm pronation, which cannot move the hand point at all) — orientation
is simply not in the budget. Its ``RobotSpec`` therefore sets
``track_orientation=False``: we keep only the 3 position rows of the jacobian
and the hand's orientation is whatever the arm geometry implies. The
orientation math below stays, used whenever a robot with ≥ 6-DoF arms is
plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from teleop import quat
from teleop.robot import Arm, Robot


@dataclass
class IKSettings:
    """Tuning knobs. Defaults are deliberately gentle; see lesson §5 to retune."""

    pos_gain: float = 12.0  # [1/s] fraction of position error to close per second
    rot_gain: float = 6.0  # [1/s] same, for orientation (unused when the robot
    #                        doesn't track orientation, e.g. the 4-DoF-arm K1)
    damping: float = 0.05  # DLS λ; larger = more stable, slower near targets
    max_joint_vel: float = 4.0  # [rad/s] per-joint speed cap (safety + realism)
    posture_gain: float = 0.5  # [1/s] nullspace pull toward the home posture


def step(robot: Robot, dt: float, settings: IKSettings | None = None) -> dict[str, float]:
    """Advance every arm one IK step toward its mocap target.

    Mutates ``robot.data.qpos`` (arm joints only) and refreshes kinematics.
    Returns {arm name: position error in meters} *before* the step, so the
    caller can display tracking quality.
    """
    settings = settings or IKSettings()
    errors: dict[str, float] = {}
    for arm in robot.arms.values():
        dq, pos_err = solve_arm_velocity(robot, arm, settings)
        q = robot.data.qpos[arm.qpos_ids] + dq * dt  # integrate one tick
        robot.data.qpos[arm.qpos_ids] = np.clip(q, arm.joint_min, arm.joint_max)
        errors[arm.name] = pos_err
    mujoco.mj_forward(robot.model, robot.data)  # recompute poses from new qpos
    return errors


def solve_arm_velocity(robot: Robot, arm: Arm, settings: IKSettings) -> tuple[np.ndarray, float]:
    """Compute joint velocities (one per arm joint) for a single arm."""
    model, data = robot.model, robot.data
    target = robot.target_pose(arm)
    current = robot.ee_pose(arm)

    # 1. Desired end-effector velocity (world frame) and 2. the jacobian,
    #    restricted to this arm's joints. With track_orientation=False (the
    #    K1: 4-DoF arms) only the 3 position rows exist — asking a 4-DoF arm
    #    to also hold an orientation would just corrupt the position tracking.
    jacp = np.zeros((3, model.nv))  # d(position)/d(joint velocities)
    jacr = np.zeros((3, model.nv))  # d(orientation)/d(joint velocities)
    mujoco.mj_jac(model, data, jacp, jacr, current.pos, arm.ee_body_id)
    if robot.spec.track_orientation:
        twist = np.empty(6)
        twist[:3] = settings.pos_gain * (target.pos - current.pos)
        twist[3:] = settings.rot_gain * quat.error_world(target.quat, current.quat)
        jac = np.vstack([jacp, jacr])[:, arm.dof_ids]  # shape (6, n_arm_joints)
    else:
        twist = settings.pos_gain * (target.pos - current.pos)
        jac = jacp[:, arm.dof_ids]  # shape (3, n_arm_joints)

    # 3. Damped least squares:  dq = Jᵀ (J Jᵀ + λ² I)⁻¹ · twist
    jjt = jac @ jac.T
    jjt[np.diag_indices_from(jjt)] += settings.damping**2
    dq = jac.T @ np.linalg.solve(jjt, twist)

    # 4. Nullspace posture task: move toward home only in ways the hand can't feel.
    #    Project the posture velocity through (I − J⁺J). The projector must use
    #    the EXACT pseudo-inverse: reusing the damped inverse leaks posture
    #    motion into task space, so the hand stops short of the target forever
    #    (measured on the K1: 0.33 mm steady-state error, vs 0.00 mm exact; the
    #    leak grows with damping and task dimension — see lesson §5).
    if settings.posture_gain > 0.0:
        dq_posture = settings.posture_gain * (arm.q_home - data.qpos[arm.qpos_ids])
        dq += dq_posture - np.linalg.pinv(jac) @ (jac @ dq_posture)

    # 5. Uniformly scale down if any joint would exceed the speed cap
    #    (uniform, so the hand's direction of motion is preserved).
    peak = float(np.max(np.abs(dq)))
    if peak > settings.max_joint_vel:
        dq *= settings.max_joint_vel / peak

    return dq, float(np.linalg.norm(target.pos - current.pos))
