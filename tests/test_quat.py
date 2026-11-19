"""Unit tests for teleop.quat (pure math, no robot model needed)."""

import math

import numpy as np

from teleop import quat

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def axis_angle(axis, angle: float) -> np.ndarray:
    """Build a wxyz quaternion rotating by `angle` around `axis` (test helper)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle
    return np.concatenate([[math.cos(half)], math.sin(half) * axis])


def test_from_axis_angle_matches_reference():
    assert np.allclose(quat.from_axis_angle([0, 0, 2.0], 0.7), axis_angle([0, 0, 1], 0.7), atol=1e-12)


def test_normalize_returns_unit_length():
    q = quat.normalize(np.array([2.0, 0.0, 0.0, 0.0]))
    assert np.allclose(q, IDENTITY)


def test_mul_identity_is_neutral():
    q = axis_angle([0, 0, 1], 0.7)
    assert np.allclose(quat.mul(q, IDENTITY), q)
    assert np.allclose(quat.mul(IDENTITY, q), q)


def test_conj_is_inverse():
    q = axis_angle([1, 2, 3], 1.1)
    assert np.allclose(quat.mul(q, quat.conj(q)), IDENTITY, atol=1e-12)


def test_from_matrix_roundtrip():
    # 90° about z as a matrix → quaternion
    mat = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(quat.from_matrix(mat), axis_angle([0, 0, 1], math.pi / 2), atol=1e-12)


def test_error_world_recovers_axis_times_angle():
    # Current = identity, target = 0.3 rad about z ⇒ error vector = [0, 0, 0.3].
    target = axis_angle([0, 0, 1], 0.3)
    err = quat.error_world(target, IDENTITY)
    assert np.allclose(err, [0.0, 0.0, 0.3], atol=1e-9)


def test_error_world_is_zero_at_target():
    q = axis_angle([1, 0, 1], 0.9)
    assert np.allclose(quat.error_world(q, q), np.zeros(3), atol=1e-12)


def test_error_world_is_expressed_in_world_frame():
    # Even if the current orientation is some arbitrary rotation, applying an
    # extra world-frame rotation about x must yield an error along world x.
    current = axis_angle([0, 1, 0], 1.2)
    extra = axis_angle([1, 0, 0], 0.4)
    target = quat.mul(extra, current)  # world-frame composition
    err = quat.error_world(target, current)
    assert np.allclose(err, [0.4, 0.0, 0.0], atol=1e-9)
