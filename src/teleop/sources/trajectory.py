"""A scripted target source: raise both hands, then trace a figure-8.

This is the first source a student runs (``uv run test-in-sim``): it proves
the whole pipeline — targets → IK → joints — before any device enters the
picture. It also doubles as a repeatable test signal (the headless smoke
test uses it).

It moves in two phases:

1. **Lift-in** (first ``ramp_time`` seconds): glide from wherever the hands
   start to a working spot in front of the chest. This matters! At the home
   pose the arms hang near the bottom of their workspace, close to a
   *singularity* — some directions of hand motion become near-impossible and
   the IK gets sluggish (lesson §5 measures this). Real manipulation happens
   in front of the chest, so that's where we draw. (The K1 tracks position
   only, so the targets keep their reset orientation throughout.)
2. **The figure-8**: a lemniscate ("∞") in the robot's y–z plane around the
   lifted center, the right hand mirroring the left:

       Δy(t) = A · sin(u)          u = 2π · (t − ramp_time) / period
       Δz(t) = A/2 · sin(2u)

Both phases are smooth and start exactly where the hands are — no jumps.
"""

from __future__ import annotations

import math

import numpy as np

from teleop.mapping import Pose
from teleop.sources import TargetFrame, TargetSource

# Where to hold the "8" relative to the home pose: forward and up, to the
# middle of the K1's workspace (measured sweep; usable arm length ~0.39 m).
LIFT_OFFSET = np.array([0.10, 0.0, 0.18])  # [m] in robot frame (x fwd, z up)


class TrajectorySource(TargetSource):
    def __init__(
        self,
        initial: TargetFrame,
        amplitude: float = 0.10,
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
        lift = smoothstep(t / self.ramp_time) * LIFT_OFFSET
        u = 2.0 * math.pi * max(0.0, t - self.ramp_time) / self.period
        dy = self.amplitude * math.sin(u)
        dz = 0.5 * self.amplitude * math.sin(2.0 * u)
        return TargetFrame(
            left=offset(self.rest_left, lift + [0.0, +dy, dz]),
            right=offset(self.rest_right, lift + [0.0, -dy, dz]),  # mirrored
        )


def smoothstep(x: float) -> float:
    """0→1 with zero velocity at both ends (the classic 3x² − 2x³)."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def offset(rest: Pose, delta: np.ndarray) -> Pose:
    """The rest pose, translated by `delta` (orientation unchanged)."""
    return Pose(rest.pos + delta, rest.quat.copy())
