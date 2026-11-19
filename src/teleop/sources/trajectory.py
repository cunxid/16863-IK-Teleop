"""A scripted target source: raise both hands, then trace a figure-8.

This is the first source a student runs (``uv run test-in-sim``): it proves
the whole pipeline — targets → IK → joints — before any device enters the
picture. It also doubles as a repeatable test signal (the headless smoke
test uses it).

It moves in two phases:

1. **Lift-in** (first ``ramp_time`` seconds): glide from wherever the hands
   start to a comfortable spot in front of the chest — position AND
   orientation. Both matter! At the rest pose the arms hang nearly straight
   down, close to a *singularity* (some hand motions become near-impossible
   and the IK gets sluggish); and keeping the hanging palm orientation at
   chest height jams the wrist-pitch joint into its limit (a measured 38°
   orientation error — lesson §5 tells both stories). So we also pitch the
   palms forward by LIFT_PITCH on the way up.
2. **Figure-8**: a lemniscate ("∞") in the robot's y–z plane around the
   lifted center, the right hand mirroring the left:

       Δy(t) = A · sin(u)          u = 2π · (t − ramp_time) / period
       Δz(t) = A/2 · sin(2u)

Both phases are smooth and start exactly where the hands are — no jumps.
"""

from __future__ import annotations

import math

import numpy as np

from teleop import quat
from teleop.mapping import Pose
from teleop.sources import TargetFrame, TargetSource

# Where to hold the "8" relative to the rest pose: forward and up, to the
# middle of the workspace (chosen for the G1; generous margins on any arm).
LIFT_OFFSET = np.array([0.22, 0.0, 0.18])  # [m] in robot frame (x fwd, z up)
# ...and pitch the palms forward on the way up (rotation about world y).
# Measured on the G1: 0.0 jams the wrist at its limit (38° residual error);
# -0.6 rad tracks with 0.0° orientation error. See lesson §5.
LIFT_PITCH = -0.6  # [rad]


class TrajectorySource(TargetSource):
    def __init__(
        self,
        initial: TargetFrame,
        amplitude: float = 0.12,
        period: float = 8.0,
        ramp_time: float = 3.0,
    ):
        assert initial.left is not None and initial.right is not None
        self.rest_left = initial.left.copy()
        self.rest_right = initial.right.copy()
        self.amplitude = amplitude  # [m] half-width of the figure-8
        self.period = period  # [s] time per loop of the 8
        self.ramp_time = ramp_time  # [s] lift-in duration

    def get_targets(self, t: float) -> TargetFrame:
        s = smoothstep(t / self.ramp_time)
        lift = s * LIFT_OFFSET
        pitch = quat.from_axis_angle([0.0, 1.0, 0.0], s * LIFT_PITCH)
        u = 2.0 * math.pi * max(0.0, t - self.ramp_time) / self.period
        dy = self.amplitude * math.sin(u)
        dz = 0.5 * self.amplitude * math.sin(2.0 * u)
        return TargetFrame(
            left=offset(self.rest_left, lift + [0.0, +dy, dz], pitch),
            right=offset(self.rest_right, lift + [0.0, -dy, dz], pitch),  # mirrored
        )


def smoothstep(x: float) -> float:
    """0→1 with zero velocity at both ends (the classic 3x² − 2x³)."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def offset(rest: Pose, delta: np.ndarray, rotation: np.ndarray) -> Pose:
    """The rest pose, translated by `delta` and rotated by `rotation` (world frame)."""
    return Pose(rest.pos + delta, quat.mul(rotation, rest.quat))
