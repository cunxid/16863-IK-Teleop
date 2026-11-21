"""Unit tests for teleop.mapping: frame conversion, recentering, the clutch."""

import math

import numpy as np

from teleop import quat
from teleop.mapping import XR_TO_MJ_MAT, Clutch, Pose, recenter_correction, xr_to_mj_pose
from tests.test_quat import IDENTITY, axis_angle


def pose(pos, q=None) -> Pose:
    return Pose(np.asarray(pos, dtype=np.float64), IDENTITY.copy() if q is None else np.asarray(q, dtype=np.float64))


# ---------------------------------------------------------------- frames ---


def test_xr_to_mj_is_a_proper_rotation():
    assert np.isclose(np.linalg.det(XR_TO_MJ_MAT), 1.0)
    assert np.allclose(XR_TO_MJ_MAT @ XR_TO_MJ_MAT.T, np.eye(3))


def test_xr_axes_map_to_expected_mj_axes():
    # XR "forward" (-z) → MJ forward (+x)
    assert np.allclose(XR_TO_MJ_MAT @ [0, 0, -1], [1, 0, 0])
    # XR "up" (+y) → MJ up (+z)
    assert np.allclose(XR_TO_MJ_MAT @ [0, 1, 0], [0, 0, 1])
    # XR "right" (+x) → MJ right (-y)
    assert np.allclose(XR_TO_MJ_MAT @ [1, 0, 0], [0, -1, 0])


def test_xr_rotation_about_up_axis_becomes_rotation_about_mj_z():
    # Turning left (about XR +y) must become turning left (about MJ +z).
    p = xr_to_mj_pose(np.zeros(3), axis_angle([0, 1, 0], 0.8))
    assert np.allclose(p.quat, axis_angle([0, 0, 1], 0.8), atol=1e-9)


# ------------------------------------------------------------- recenter ---


def test_recenter_identity_when_already_facing_forward():
    correction = recenter_correction(IDENTITY)
    assert np.allclose(quat.rotate(correction, [0, 0, -1]), [0, 0, -1], atol=1e-9)


def test_recenter_makes_the_users_forward_the_xr_forward():
    # Head turned 90° left (about XR up): the user's "push forward" is XR -x.
    head = axis_angle([0, 1, 0], math.pi / 2)
    correction = recenter_correction(head)
    # Corrected, that push must be XR -z (which the fixed map sends to MJ +x).
    assert np.allclose(quat.rotate(correction, [-0.1, 0, 0]), [0, 0, -0.1], atol=1e-9)
    assert np.allclose(XR_TO_MJ_MAT @ quat.rotate(correction, [-0.1, 0, 0]), [0.1, 0, 0], atol=1e-9)


def test_recenter_corrects_yaw_only_up_stays_up():
    # Turned 90° left AND looking 30° down: the yaw fix must not tilt the world.
    head = quat.mul(axis_angle([0, 1, 0], math.pi / 2), axis_angle([1, 0, 0], -0.5))
    correction = recenter_correction(head)
    assert np.allclose(quat.rotate(correction, [0, 1, 0]), [0, 1, 0], atol=1e-9)  # up
    assert np.allclose(quat.rotate(correction, [-1, 0, 0]), [0, 0, -1], atol=1e-9)  # fwd


def test_recenter_refuses_a_vertical_gaze():
    looking_straight_down = axis_angle([1, 0, 0], -math.pi / 2)
    assert recenter_correction(looking_straight_down) is None


# ---------------------------------------------------------------- clutch ---


def test_clutch_released_means_frozen():
    c = Clutch(pose([1, 2, 3]))
    out = c.update(grip_held=False, controller=pose([9, 9, 9]))
    assert np.allclose(out.pos, [1, 2, 3])
    assert not c.engaged


def test_clutch_translates_by_scaled_delta():
    c = Clutch(pose([1.0, 0.0, 0.0]), scale=0.5)
    c.update(True, pose([0.0, 0.0, 0.0]))  # grab (snapshot, no motion yet)
    out = c.update(True, pose([0.2, 0.0, -0.4]))  # move the controller
    assert np.allclose(out.pos, [1.0 + 0.1, 0.0, -0.2])


def test_clutch_ignores_controller_rotation():
    """Translation-only: wild wrist rotation must never change the target quat."""
    q0 = axis_angle([0, 0, 1], 0.5)
    c = Clutch(pose([0, 0, 0], q0))
    c.update(True, pose([0, 0, 0]))
    out = c.update(True, pose([0.1, 0, 0], axis_angle([1, 0, 0], 0.3)))
    assert np.allclose(out.quat, q0, atol=1e-9)  # orientation kept
    assert np.allclose(out.pos, [0.1, 0, 0])  # translation still applies


def test_clutch_no_jump_on_regrab():
    """The whole point of a clutch: repositioning while released is invisible."""
    c = Clutch(pose([0.0, 0.0, 0.0]))
    c.update(True, pose([0.0, 0.0, 0.0]))
    c.update(True, pose([0.1, 0.0, 0.0]))  # target now at x=0.1
    c.update(False, None)  # release
    out = c.update(True, pose([5.0, 5.0, 5.0]))  # re-grab from a far-away arm pose
    assert np.allclose(out.pos, [0.1, 0.0, 0.0])  # no jump
    out = c.update(True, pose([5.0, 5.0, 5.2]))  # further deltas apply again
    assert np.allclose(out.pos, [0.1, 0.0, 0.2])


