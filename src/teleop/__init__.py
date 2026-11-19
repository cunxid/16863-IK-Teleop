"""ik-teleop: a teaching codebase for VR teleoperation of a humanoid upper body.

Pipeline (each stage is one small module):

    TargetSource ──▶ mocap targets ──▶ differential IK ──▶ joint angles ──▶ sim viewer / real robot
    (sources/)       (robot.py)        (ik.py)             (robot.py)       (run_sim.py / run_real.py)

Start reading at ``lesson/index.html`` — it walks through every file in the
order a student should learn them.
"""

from pathlib import Path

__version__ = "0.1.0"

# The repository root (this file lives at <root>/src/teleop/__init__.py).
# Assets and the lesson are addressed relative to it, so every command must be
# run from a checkout of the repo (this is a teaching repo, not a library).
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
