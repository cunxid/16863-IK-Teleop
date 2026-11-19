"""Quaternion helpers — the only math utilities in this project.

Conventions used EVERYWHERE in this codebase:

* A quaternion is a numpy array of 4 floats in **wxyz order** (w first).
  This is MuJoCo's native order. Careful: WebXR and SciPy use xyzw!
* A unit quaternion represents a rotation. Composing two rotations is
  quaternion multiplication; "undoing" a unit rotation is its conjugate.

We delegate the actual arithmetic to MuJoCo's well-tested ``mju_*`` C
functions and only wrap them in a friendlier, allocation-per-call API.
(MuJoCo's functions write results into output arrays you pass in; these
wrappers hide that so the rest of the code reads like plain math.)
"""

from __future__ import annotations

import mujoco
import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    """Return q scaled to unit length (a valid rotation)."""
    out = np.asarray(q, dtype=np.float64).copy()
    mujoco.mju_normalize4(out)
    return out


def mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compose rotations: first apply q2, then q1 (like matrix product R1 @ R2)."""
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(q1, dtype=np.float64), np.asarray(q2, dtype=np.float64))
    return out


def conj(q: np.ndarray) -> np.ndarray:
    """Conjugate = inverse rotation (for unit quaternions): mul(q, conj(q)) == identity."""
    out = np.empty(4)
    mujoco.mju_negQuat(out, np.asarray(q, dtype=np.float64))
    return out


def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation of `angle` radians around `axis` (need not be unit length)."""
    out = np.empty(4)
    unit = np.asarray(axis, dtype=np.float64).copy()
    mujoco.mju_normalize3(unit)
    mujoco.mju_axisAngle2Quat(out, unit, angle)
    return out


def from_matrix(mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a (wxyz) quaternion."""
    out = np.empty(4)
    mujoco.mju_mat2Quat(out, np.asarray(mat, dtype=np.float64).reshape(9))
    return out


def error_world(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) that takes q_current onto q_target.

    Returns axis * angle, in radians, shape (3,). This is the "orientation
    error" used by the IK: if the end-effector spins with this angular
    velocity for one second, its orientation becomes the target.

    Math: the world-frame error rotation is q_err = q_target ⊗ q_current⁻¹
    (apply it *after* the current rotation and you land on the target);
    ``mju_quat2Vel`` is its log map (quaternion → rotation vector).
    """
    q_err = mul(q_target, conj(q_current))
    out = np.empty(3)
    mujoco.mju_quat2Vel(out, q_err, 1.0)
    return out
