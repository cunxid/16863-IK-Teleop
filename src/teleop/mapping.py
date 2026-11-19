"""VR-space → robot-space mapping, and the clutch.

Two small, pure-math pieces live here (no MuJoCo state, fully unit-testable):

1. **Frame conversion.** WebXR and MuJoCo disagree on axes:

       WebXR  (headset):  x = right,  y = up,    z = backward  (-z = where you face)
       MuJoCo (robot):    x = forward, y = left,  z = up

   Standing *behind* the robot, facing the same way it faces, "push the
   controller forward" must become "+x in robot world". That is a fixed
   rotation, applied to every controller pose as it enters the system.

2. **The clutch** (like a mouse you can lift). Absolute mapping would need
   calibration ("where is the headset relative to the robot?"). Instead we
   use *relative* mapping: while you HOLD the grip button, controller
   deltas are applied to the target; release, reposition your arm
   comfortably, grab again — the target never jumps. This is the single
   most important UX trick in VR teleoperation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from teleop import quat


@dataclass
class Pose:
    """A position + orientation in the MuJoCo world frame.

    pos:  shape (3,), meters.
    quat: shape (4,), wxyz unit quaternion.
    """

    pos: np.ndarray
    quat: np.ndarray

    def copy(self) -> "Pose":
        return Pose(self.pos.copy(), self.quat.copy())


# Rows map XR axes onto MuJoCo axes:  x_mj = -z_xr,  y_mj = -x_xr,  z_mj = +y_xr.
# (A proper rotation: det = +1. Verified in tests/test_mapping.py.)
XR_TO_MJ_MAT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
XR_TO_MJ_QUAT = quat.from_matrix(XR_TO_MJ_MAT)


def xr_to_mj_pose(pos_xr: np.ndarray, quat_xr_wxyz: np.ndarray) -> Pose:
    """Convert a WebXR pose (already reordered to wxyz!) into MuJoCo axes.

    Positions rotate by the fixed matrix. Orientations conjugate by the same
    rotation R: R_mj = R_fix · R_xr · R_fixᵀ, i.e. "the same physical rotation,
    expressed in the robot's axes".
    """
    pos_mj = XR_TO_MJ_MAT @ np.asarray(pos_xr, dtype=np.float64)
    q = quat.mul(XR_TO_MJ_QUAT, quat.mul(np.asarray(quat_xr_wxyz, dtype=np.float64), quat.conj(XR_TO_MJ_QUAT)))
    return Pose(pos_mj, quat.normalize(q))


class Clutch:
    """Relative (grab-and-drag) mapping from one controller to one target pose.

    State machine, one instance per hand:

        released ── grip pressed ──▶ engaged (snapshot controller & target)
        engaged  ── grip released ─▶ released (target freezes where it is)

    While engaged, every update applies the controller's *delta since the
    snapshot* to the target's snapshot:

        target.pos  = target₀.pos + scale · (controller.pos − controller₀.pos)
        target.quat = (controller.quat ⊗ controller₀.quat⁻¹) ⊗ target₀.quat
                       └── world-frame rotation since grab ──┘
    """

    def __init__(self, initial_target: Pose, scale: float = 1.0):
        self.scale = scale
        self._target = initial_target.copy()
        self._engaged = False
        self._controller0: Pose | None = None
        self._target0: Pose | None = None

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def target(self) -> Pose:
        """Current target pose (frozen while the grip is released)."""
        return self._target.copy()

    def update(self, grip_held: bool, controller: Pose | None) -> Pose:
        """Feed one controller sample; returns the (possibly moved) target."""
        if not grip_held or controller is None:
            self._engaged = False
            return self.target

        if not self._engaged:  # grip just pressed: take snapshots
            self._engaged = True
            self._controller0 = controller.copy()
            self._target0 = self._target.copy()

        assert self._controller0 is not None and self._target0 is not None
        delta_pos = self.scale * (controller.pos - self._controller0.pos)
        delta_rot = quat.mul(controller.quat, quat.conj(self._controller0.quat))
        self._target = Pose(
            self._target0.pos + delta_pos,
            quat.normalize(quat.mul(delta_rot, self._target0.quat)),
        )
        return self.target
