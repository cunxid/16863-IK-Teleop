"""The robot: load the model, name its parts, own the targets.

This is the only file that knows robot-specific strings (joint names, body
names, home pose). Everything else talks to a ``Robot`` through poses and
joint arrays, so each robot is described by exactly one ``RobotSpec``.

Two robots are specced here, both from Booster Robotics' official MuJoCo
models in github.com/BoosterRobotics/booster_assets (BSD-3-Clause;
``uv run setup-assets`` downloads them):

* the **Booster K1** (0.95 m, 19.7 kg; 22 joints: 2 head + 4 per arm +
  6 per leg — no waist, no wrists, no hands), the default, and
* the **Booster T1** (1.2 m, ~30 kg; 23 joints: the K1 layout plus a waist
  yaw joint — same 4-DoF arms, no wrists, no hands).

Facts that shaped these specs (all measured, see lesson §2 and §5):

* **4-DoF arms** (both robots) ⇒ a full 6-D pose (position + orientation) is
  out of the DoF budget, so ``track_orientation=False``: the IK tracks hand
  *position* only.
* **Names are copied verbatim**, including the vendor's odd ``aa`` prefixes —
  and the two robots disagree about where those go: on the K1 the shoulder
  and head joints carry them (``aaleft_shoulder_pitch_joint``), on the T1
  only the head joints do (``aahead_yaw_joint`` but plain
  ``left_shoulder_pitch_joint``). Never "fix" vendor names. Also, names can
  lie about function: on BOTH robots, ``elbow_pitch`` is forearm *pronation*
  (spins the forearm about its own axis — the hand point doesn't move at all)
  and ``elbow_yaw`` is the actual elbow *bend*.
* **No keyframe in either model**, and qpos=0 is a flying T-pose — so the
  home pose (trunk height for feet-on-floor + relaxed arm angles) is authored
  right here in the spec.

One MuJoCo indexing trap worth learning early (lesson §2): a joint has TWO
indices. For the K1, ``nq=29`` but ``nv=28``, because the floating base takes
7 qpos slots (3 position + 4 quaternion) but only 6 velocity slots. Hence each
arm stores both ``qpos_ids`` (joint angles) and ``dof_ids`` (jacobian
columns) — mixing them up is the classic off-by-one bug in MuJoCo code.
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
    joint_names: tuple[str, ...]  # kinematic order, shoulder → elbow
    ee_body: str  # last arm body (the forearm + fixed hand cap)
    ee_offset: tuple[float, float, float]  # palm center, in that body's frame [m]
    mocap_body: str  # the draggable target cube in our scene XML


@dataclass(frozen=True)
class RobotSpec:
    name: str
    scene_file: str  # our teleop scene, relative to assets/ (installed by setup-assets)
    scene_template: str  # the scene source in assets/, copied next to the vendor model
    vendor_dir: str  # folder inside the booster_assets repo, e.g. "robots/K1"
    model_file: str  # the vendor MJCF the scene includes, e.g. "K1_22dof.xml"
    arms: tuple[ArmSpec, ...]
    base_joint: str  # the floating base (freejoint) name
    home_base_height: float  # [m] trunk height that puts the feet on the floor
    home_joint_angles: tuple[tuple[str, float], ...]  # authored home pose [rad]
    track_orientation: bool  # False for arms with < 6 DoF (lesson §5)

    def scene_path(self) -> Path:
        return ASSETS_DIR / self.scene_file

    def model_dir(self) -> Path:
        """Where setup-assets installs this robot's vendor model."""
        return self.scene_path().parent

    def model_path(self) -> Path:
        return self.model_dir() / self.model_file


# Booster K1 — all numbers measured on booster_assets K1_22dof.xml.
K1 = RobotSpec(
    name="k1",
    scene_file="booster_k1/scene_teleop.xml",
    scene_template="scene_teleop_k1.xml",
    vendor_dir="robots/K1",
    model_file="K1_22dof.xml",
    base_joint="world_joint",
    home_base_height=0.552,  # feet exactly on the floor (measured)
    # qpos=0 is a T-pose; this hangs the arms at the sides, slightly forward,
    # with a relaxed elbow bend. Signs mirror (elbow_yaw bends "forward" with
    # opposite signs on the two arms — the ranges in the XML mirror too).
    home_joint_angles=(
        ("aaleft_shoulder_pitch_joint", -0.25),
        ("left_shoulder_roll_joint", -1.40),
        ("left_elbow_yaw_joint", -0.60),
        ("aaright_shoulder_pitch_joint", -0.25),
        ("right_shoulder_roll_joint", 1.40),
        ("right_elbow_yaw_joint", 0.60),
    ),
    track_orientation=False,  # 4-DoF arms: position-only tracking
    arms=(
        ArmSpec(
            name="left",
            joint_names=(
                "aaleft_shoulder_pitch_joint",  # vendor name, 'aa' and all
                "left_shoulder_roll_joint",
                "left_elbow_pitch_joint",  # forearm pronation (nullspace joint)
                "left_elbow_yaw_joint",  # the actual elbow bend
            ),
            ee_body="left_elbow_yaw_link",
            ee_offset=(0.0, 0.18, 0.0),  # palm center: forearm runs along +y
            mocap_body="left_target",
        ),
        ArmSpec(
            name="right",
            joint_names=(
                "aaright_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_elbow_pitch_joint",
                "right_elbow_yaw_joint",
            ),
            ee_body="right_elbow_yaw_link",
            ee_offset=(0.0, -0.18, 0.0),  # mirrored: forearm runs along -y
            mocap_body="right_target",
        ),
    ),
)

# Booster T1 — all numbers measured on booster_assets T1_23dof.xml. Same
# layout as the K1 (4-DoF arms, elbow_pitch = pronation, elbow_yaw = bend,
# no keyframe) with one extra waist yaw joint we leave at 0, and shoulder
# joints WITHOUT the K1's 'aa' prefix (only the T1 head joints carry it).
T1 = RobotSpec(
    name="t1",
    scene_file="booster_t1/scene_teleop.xml",
    scene_template="scene_teleop_t1.xml",
    vendor_dir="robots/T1",
    model_file="T1_23dof.xml",
    base_joint="world_joint",
    home_base_height=0.687,  # feet exactly on the floor (measured)
    # Same relaxed pose as the K1 — the sign conventions transfer verbatim
    # (measured: palms land symmetric at x=+0.27 m, z=0.57 m).
    home_joint_angles=(
        ("left_shoulder_pitch_joint", -0.25),
        ("left_shoulder_roll_joint", -1.40),
        ("left_elbow_yaw_joint", -0.60),
        ("right_shoulder_pitch_joint", -0.25),
        ("right_shoulder_roll_joint", 1.40),
        ("right_elbow_yaw_joint", 0.60),
    ),
    track_orientation=False,  # 4-DoF arms, like the K1: position-only tracking
    arms=(
        ArmSpec(
            name="left",
            joint_names=(
                "left_shoulder_pitch_joint",  # no 'aa' prefix on the T1
                "left_shoulder_roll_joint",
                "left_elbow_pitch_joint",  # forearm pronation (nullspace joint)
                "left_elbow_yaw_joint",  # the actual elbow bend
            ),
            ee_body="left_elbow_yaw_link",
            ee_offset=(0.0, 0.216, 0.0),  # hand-cap sphere center (measured)
            mocap_body="left_target",
        ),
        ArmSpec(
            name="right",
            joint_names=(
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_elbow_pitch_joint",
                "right_elbow_yaw_joint",
            ),
            ee_body="right_elbow_yaw_link",
            ee_offset=(0.0, -0.216, 0.0),  # mirrored: forearm runs along -y
            mocap_body="right_target",
        ),
    ),
)

SPECS: dict[str, RobotSpec] = {"k1": K1, "t1": T1}


@dataclass
class Arm:
    """One arm with every name resolved to fast numeric indices."""

    name: str
    joint_names: tuple[str, ...]
    qpos_ids: np.ndarray  # indices into data.qpos (joint angles)
    dof_ids: np.ndarray  # indices into velocity space (jacobian columns)
    joint_min: np.ndarray  # [rad] lower limits, same order as joint_names
    joint_max: np.ndarray  # [rad] upper limits
    q_home: np.ndarray  # [rad] authored home posture
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
                f"Download it first:  uv run setup-assets --robot {spec.name}"
            )
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.arms: dict[str, Arm] = {a.name: self._resolve_arm(a) for a in spec.arms}

    def _resolve_arm(self, spec: ArmSpec) -> Arm:
        joint_ids = [self.model.joint(name).id for name in spec.joint_names]
        home = dict(self.spec.home_joint_angles)
        ranges = self.model.jnt_range[joint_ids]  # shape (n, 2): [min, max]
        return Arm(
            name=spec.name,
            joint_names=spec.joint_names,
            qpos_ids=np.array([self.model.jnt_qposadr[j] for j in joint_ids]),
            dof_ids=np.array([self.model.jnt_dofadr[j] for j in joint_ids]),
            joint_min=ranges[:, 0].copy(),
            joint_max=ranges[:, 1].copy(),
            q_home=np.array([home.get(name, 0.0) for name in spec.joint_names]),
            ee_body_id=self.model.body(spec.ee_body).id,
            ee_offset=np.array(spec.ee_offset),
            mocap_id=int(self.model.body(spec.mocap_body).mocapid[0]),
        )

    # ------------------------------------------------------------- state ---

    def reset(self) -> None:
        """Authored home pose, then park each target exactly on its hand.

        Neither Booster model ships a keyframe, so "home" is built by hand: default
        state, base lowered until the feet touch the floor, then the authored
        joint angles (everything unnamed stays 0 — the legs are straight).
        """
        mujoco.mj_resetData(self.model, self.data)
        base_adr = self.model.jnt_qposadr[self.model.joint(self.spec.base_joint).id]
        self.data.qpos[base_adr + 2] = self.spec.home_base_height  # z of the trunk
        for name, angle in self.spec.home_joint_angles:
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = angle
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

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Arm joint limits [rad] by name — the real driver re-clamps to these
        (safety layer 4, lesson §9: a driver never trusts its caller)."""
        out: dict[str, tuple[float, float]] = {}
        for arm in self.arms.values():
            for name, lo, hi in zip(arm.joint_names, arm.joint_min, arm.joint_max):
                out[name] = (float(lo), float(hi))
        return out
