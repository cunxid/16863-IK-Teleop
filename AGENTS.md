# AGENTS.md — rules for anyone (human or AI) editing this repo

## What this project is

A **teaching codebase** that teleoperates the upper body of the **Booster
K1** humanoid with PICO VR controllers: you move the controller, the robot's
hand follows, via differential inverse kinematics (IK). MuJoCo is the
simulation backend; the same IK output can be streamed to the real robot.
The K1's arms are 4-DoF, so the IK tracks hand *position* only
(`RobotSpec.track_orientation = False`).

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
  real_robot.py   # RealRobot interface + DryRunRobot (default) + robot factory
  booster_k1.py   # real K1 driver: robot stands/balances itself (UpperBodyCustomControl), we command only the arms
  run_sim.py      # entry point: test-in-sim
  run_real.py     # entry point: test-in-real
  tools/          # setup_assets (download robot model), check_lesson
```

## Commands

```bash
uv sync                       # install everything (incl. dev group)
uv run setup-assets           # download the robot model (Booster K1, booster_assets)
uv run test-in-sim            # demo: scripted trajectory in the MuJoCo viewer
uv run test-in-sim --source mouse   # drag the red/green target cubes yourself
uv run test-in-sim --source pico    # VR: open the printed URL in the Pico browser
uv run test-in-sim --headless       # no GUI; prints tracking error (CI/servers)
uv run test-in-real           # stream IK joint targets to a robot (default: dry-run)
uv sync --extra real          # add Booster's SDK (hardware only; Linux wheels)
uv run test-in-real --robot k1 --stand   # real K1: stand up, balance, then VR-drive the arms
uv run pytest                 # unit tests
uv run check-lesson           # enforce the Lesson Rule
uv run render-videos          # demo MP4s into video/ (offscreen, EGL; no display)
```

## Robot model policy

- The robot is the **Booster K1**, using Booster Robotics' official MuJoCo
  model from `BoosterRobotics/booster_assets` (BSD-3-Clause) — sim and real
  are the same machine. It is downloaded by `uv run setup-assets` into
  `assets/booster_k1/` and is **never committed** (see .gitignore).
- K1 names are copied **verbatim**, including the vendor's `aa` prefixes
  (`aaleft_shoulder_pitch_joint`); never "normalize" them. The model ships
  no keyframe — the home pose is authored in the `RobotSpec`.
- Everything robot-specific lives in **one `RobotSpec` object** in
  `src/teleop/robot.py` (XML path, arm joint names, end-effector bodies,
  home pose, whether orientation is tracked). To use another robot: add a
  `RobotSpec` and pass `--robot-spec`. No other file may hard-code robot
  names.

## Real-robot safety invariants (never remove these)

`run_real.py` must always keep: joint-limit clamping, per-tick joint velocity
clamping, a deadman (targets only move while the grip button is held), and a
watchdog (stale VR data ⇒ hold position). It must also keep its argparse
gates: deadman-free sources (anything but `pico`) are allowed only with
`--robot dryrun`, and `--rate` values in the IK's unstable zone (gain·dt ≥ 2)
are rejected, and `--stand` (a physical stand-up) requires `--robot k1`. A
print-only robot (`DryRunRobot`) is the default so `test-in-real` is always
runnable without hardware. The K1 driver (`booster_k1.py`) adds its own
invariants: it re-clamps targets to joint limits (never trust the caller),
soft-starts by blending from the measured arm pose, commands only the 8 arm
motors (zero gains elsewhere ⇒ the onboard controller keeps balancing), and
on `disconnect()` hands the arms back via `UpperBodyCustomControl(False)` —
never damping mode, which would drop a standing robot.

## Testing

- `tests/` holds fast, headless unit tests (no viewer; the only network use
  is the localhost-only Pico bridge test).
- IK and mapping math must stay covered: if you touch `ik.py`, `quat.py` or
  `mapping.py`, run `uv run pytest` and extend tests for new behavior.
- `uv run test-in-sim --headless` is the end-to-end smoke test; it exits
  non-zero if tracking error stays above threshold.

## Housekeeping

- `uv` manages everything; commit `uv.lock`. Never `pip install` into the venv
  directly. Add deps with `uv add` (or `uv add --group dev` for tooling).
- This machine may have no display: never launch the viewer in automated
  checks; use `--headless`.
