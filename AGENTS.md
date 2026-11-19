# AGENTS.md — rules for anyone (human or AI) editing this repo

## What this project is

A **teaching codebase** that teleoperates the upper body of a humanoid
robot with PICO VR controllers: you move the controller, the robot's hand
follows, via differential inverse kinematics (IK). MuJoCo is the simulation
backend; the same IK output can be streamed to a real robot.

The audience is **students with no robotics background**. Every design
decision trades cleverness for readability.

## ★ The Lesson Rule — most important rule in this file

All code in this repo must map to **one HTML file**: [`lesson/index.html`](lesson/index.html).
That file is the lesson plan: a step-by-step walkthrough of how a student
builds and understands this system, from an empty folder to VR teleoperation.

Concretely:

1. Every source file (`src/teleop/**`, `assets/*.xml`, `pyproject.toml`) must be
   covered by a section of `lesson/index.html` that explains what it does and why.
2. **Any code change must update the lesson in the same change.** New file → new
   lesson section. Changed behavior → changed explanation and code excerpts.
3. `uv run check-lesson` verifies the file↔lesson mapping and must pass before
   a change is considered done.
4. Everything is written in English: the lesson, README, UI text, code
   identifiers, comments and docstrings. The lesson is self-contained
   (no external JS/CSS).
5. The lesson teaches in *build order*, not file order: environment → MuJoCo
   model → forward kinematics → targets (mocap) → differential IK → sim loop →
   trajectory source → VR source → real robot.

## Code style for teaching

- Clarity beats performance and beats abstraction. No metaprogramming, no
  decorators beyond `@dataclass`, no inheritance deeper than one level.
- Every module starts with a docstring saying **where it sits in the pipeline**.
- Type hints everywhere; `numpy` arrays documented with shape and convention
  (quaternions are **wxyz, MuJoCo order**; positions are meters, world frame).
- Keep modules small (aim < ~200 lines). If a file grows past that, split it
  and add a lesson section for the new file.
- Core dependencies stay minimal: `mujoco`, `numpy`, `websockets`,
  `cryptography`. Do not add a dependency when 20 lines of stdlib will do.
- Explicit loops over clever vectorization when the loop is clearer.

## Architecture (one line per stage)

```
TargetSource (trajectory | mouse | pico) ──writes──▶ mocap target bodies (in MuJoCo data)
mocap targets ──tracked by──▶ differential IK (ik.py, damped least squares)
IK ──updates──▶ data.qpos of the arm joints (kinematic sim: mj_forward, no mj_step)
qpos ──shown in──▶ MuJoCo viewer (test-in-sim)   and/or   streamed to ──▶ RealRobot (test-in-real)
```

The mocap bodies are the single source of truth for "where should the hands
be". Sources write them, the mouse drags them, IK only ever reads them.

```
src/teleop/
  robot.py        # load model, resolve joint/body ids, reset, read/write targets
  quat.py         # quaternion helpers (wxyz) — the only math utilities
  ik.py           # differential IK: pose error → joint velocities → qpos
  mapping.py      # VR→robot frame conversion + clutch (relative mapping)
  sources/        # TargetSource implementations: trajectory, mouse, pico (WebXR)
  real_robot.py   # RealRobot interface + DryRunRobot + FaunaRobot stub
  run_sim.py      # entry point: test-in-sim
  run_real.py     # entry point: test-in-real
  tools/          # setup_assets (download robot model), check_lesson
```

## Commands

```bash
uv sync                       # install everything (incl. dev group)
uv run setup-assets           # download the robot model (mujoco_menagerie G1)
uv run test-in-sim            # demo: scripted trajectory in the MuJoCo viewer
uv run test-in-sim --source mouse   # drag the red/green target cubes yourself
uv run test-in-sim --source pico    # VR: open the printed URL in the Pico browser
uv run test-in-sim --headless       # no GUI; prints tracking error (CI/servers)
uv run test-in-real           # stream IK joint targets to a robot (default: dry-run)
uv run pytest                 # unit tests
uv run check-lesson           # enforce the Lesson Rule
```

## Robot model policy

- The teleop target is Fauna Robotics' **"Sprout"** (6-DoF arms, 1-DoF
  grippers); its URDF/MJCF exists but ships only inside Fauna's login-gated
  SDK, so the demo uses the **Unitree G1** from
  `google-deepmind/mujoco_menagerie` (BSD-3, 7-DoF arms) as a stand-in with
  the same dual-arm humanoid topology. It is downloaded by
  `uv run setup-assets` into `assets/unitree_g1/` and is **never committed**
  (see .gitignore).
- Everything robot-specific lives in **one `RobotSpec` object** in
  `src/teleop/robot.py` (XML path, arm joint names, end-effector bodies,
  keyframe name). To swap in the real Fauna model: add a `RobotSpec` for it
  and pass `--robot-spec`. No other file may hard-code robot names.

## Real-robot safety invariants (never remove these)

`run_real.py` must always keep: joint-limit clamping, per-tick joint velocity
clamping, a deadman (targets only move while the grip button is held), and a
watchdog (stale VR data ⇒ hold position). A stub robot (`DryRunRobot`) is the
default so `test-in-real` is always runnable without hardware.

## Testing

- `tests/` holds fast, headless unit tests (no viewer, no network).
- IK and mapping math must stay covered: if you touch `ik.py`, `quat.py` or
  `mapping.py`, run `uv run pytest` and extend tests for new behavior.
- `uv run test-in-sim --headless` is the end-to-end smoke test; it exits
  non-zero if tracking error stays above threshold.

## Housekeeping

- `uv` manages everything; commit `uv.lock`. Never `pip install` into the venv
  directly. Add deps with `uv add` (or `uv add --group dev` for tooling).
- This machine may have no display: never launch the viewer in automated
  checks; use `--headless`.
