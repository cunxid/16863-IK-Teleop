"""The robot: load the model, name its parts, own the targets.

This is the only file that knows robot-specific strings (joint names, body
names, keyframes). Everything else talks to a ``Robot`` through poses and
joint arrays, so swapping robots = adding one ``RobotSpec`` below.

About the demo model: the teleop target is Fauna Robotics' "Sprout"
(6-DoF arms + 1-DoF grippers), whose URDF/MJCF ships only inside their gated
SDK. As a public stand-in with the same dual-arm humanoid topology we use the
Unitree G1 (7-DoF arms) from mujoco_menagerie — downloaded, not committed;
run ``uv run setup-assets`` once.

One MuJoCo indexing trap worth learning early (lesson §2): a joint has TWO
indices. ``qpos`` (positions) and the velocity/jacobian space ``nv`` differ
because the floating base takes 7 qpos (3 pos + 4 quat) but only 6 velocity
slots. Hence each arm stores both ``qpos_ids`` and ``dof_ids`` — mixing them
up is the classic off-by-one bug in MuJoCo code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from teleop import ASSETS_DIR
from teleop.mapping import Pose
from teleop.sources import TargetFrame


@dataclass(frozen=True)
class ArmSpec:
    """Names describing one arm, exactly as written in the model XML."""

    name: str  # "left" | "right"
    joint_names: tuple[str, ...]  # kinematic order, shoulder → wrist
    ee_body: str  # body the hand is welded to
    ee_offset: tuple[float, float, float]  # palm center, in that body's frame [m]
    mocap_body: str  # the draggable target cube in our scene XML


@dataclass(frozen=True)
class RobotSpec:
    name: str
    scene_file: str  # relative to assets/
    arms: tuple[ArmSpec, ...]
    home_keyframe: str

    def scene_path(self) -> Path:
        return ASSETS_DIR / self.scene_file


# Unitree G1 (stand-in for Fauna "Sprout"; see module docstring).
# Joint names from mujoco_menagerie g1.xml; palm offset (0.08, 0, 0) in the
# wrist_yaw link is the menagerie convention (g1_mjx.xml's palm sites).
G1 = RobotSpec(
    name="g1",
    scene_file="unitree_g1/scene_teleop.xml",
    home_keyframe="stand",
    arms=(
        ArmSpec(
            name="left",
            joint_names=(
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ),
            ee_body="left_wrist_yaw_link",
            ee_offset=(0.08, 0.0, 0.0),
            mocap_body="left_target",
        ),
        ArmSpec(
            name="right",
            joint_names=(
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ),
            ee_body="right_wrist_yaw_link",
            ee_offset=(0.08, 0.0, 0.0),
            mocap_body="right_target",
        ),
    ),
)

SPECS: dict[str, RobotSpec] = {"g1": G1}


@dataclass
class Arm:
    """One arm with every name resolved to fast numeric indices."""

    name: str
    joint_names: tuple[str, ...]
    qpos_ids: np.ndarray  # indices into data.qpos (joint angles)
    dof_ids: np.ndarray  # indices into velocity space (jacobian columns)
    joint_min: np.ndarray  # [rad] lower limits, same order as joint_names
    joint_max: np.ndarray  # [rad] upper limits
    q_home: np.ndarray  # [rad] posture from the home keyframe
    ee_body_id: int
    ee_offset: np.ndarray  # palm center in the ee body frame
    mocap_id: int  # row of data.mocap_pos / data.mocap_quat


class Robot:
    """MuJoCo model + data + resolved arms + the mocap targets."""

    def __init__(self, spec: RobotSpec):
        path = spec.scene_path()
        if not path.exists():
            raise FileNotFoundError(
                f"robot model not found: {path}\n"
                f"Download it first:  uv run setup-assets"
            )
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.arms: dict[str, Arm] = {a.name: self._resolve_arm(a) for a in spec.arms}

    def _resolve_arm(self, spec: ArmSpec) -> Arm:
        joint_ids = [self.model.joint(name).id for name in spec.joint_names]
        qpos_ids = np.array([self.model.jnt_qposadr[j] for j in joint_ids])
        dof_ids = np.array([self.model.jnt_dofadr[j] for j in joint_ids])
        ranges = self.model.jnt_range[joint_ids]  # shape (n, 2): [min, max]
        key = self.model.key(self.spec.home_keyframe)
        return Arm(
            name=spec.name,
            joint_names=spec.joint_names,
            qpos_ids=qpos_ids,
            dof_ids=dof_ids,
            joint_min=ranges[:, 0].copy(),
            joint_max=ranges[:, 1].copy(),
            q_home=key.qpos[qpos_ids].copy(),
            ee_body_id=self.model.body(spec.ee_body).id,
            ee_offset=np.array(spec.ee_offset),
            mocap_id=int(self.model.body(spec.mocap_body).mocapid[0]),
        )

    # ------------------------------------------------------------- state ---

    def reset(self) -> None:
        """Home keyframe, then park each target exactly on its hand."""
        key_id = self.model.key(self.spec.home_keyframe).id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)  # compute world poses from qpos
        for arm in self.arms.values():
            self.set_target(arm, self.ee_pose(arm))

    def ee_pose(self, arm: Arm) -> Pose:
        """Where the palm is right now (world frame)."""
        body = self.data.body(arm.ee_body_id)
        rotation = body.xmat.reshape(3, 3)
        return Pose(body.xpos + rotation @ arm.ee_offset, body.xquat.copy())

    # ------------------------------------------------------------ targets ---

    def target_pose(self, arm: Arm) -> Pose:
        """Where the palm *should* be (reads the mocap body)."""
        return Pose(
            self.data.mocap_pos[arm.mocap_id].copy(),
            self.data.mocap_quat[arm.mocap_id].copy(),
        )

    def set_target(self, arm: Arm, pose: Pose) -> None:
        self.data.mocap_pos[arm.mocap_id] = pose.pos
        self.data.mocap_quat[arm.mocap_id] = pose.quat

    def targets(self) -> TargetFrame:
        return TargetFrame(
            left=self.target_pose(self.arms["left"]),
            right=self.target_pose(self.arms["right"]),
        )

    def set_targets(self, frame: TargetFrame) -> None:
        if frame.left is not None:
            self.set_target(self.arms["left"], frame.left)
        if frame.right is not None:
            self.set_target(self.arms["right"], frame.right)

    # --------------------------------------------------------- for "real" ---

    def joint_targets(self) -> dict[str, float]:
        """Current arm joint angles, by name — what we stream to a real robot."""
        out: dict[str, float] = {}
        for arm in self.arms.values():
            for name, qpos_id in zip(arm.joint_names, arm.qpos_ids):
                out[name] = float(self.data.qpos[qpos_id])
        return out
