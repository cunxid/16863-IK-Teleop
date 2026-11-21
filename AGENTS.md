# AGENTS.md — rules for anyone (human or AI) editing this repo

## What this project is

A **teaching codebase** that teleoperates the upper body of a **Booster
humanoid** (the K1 by default; the taller T1 is equally supported) with PICO
VR controllers: you move the controller, the robot's hand follows, via
differential inverse kinematics (IK). MuJoCo is the simulation backend; the
same IK output can be streamed to the real robot. Both robots' arms are
4-DoF, so the IK tracks hand *position* only
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
  robot.py        # load model, resolve joint/body ids, reset, read/write targets; one RobotSpec per robot (k1, t1)
  quat.py         # quaternion helpers (wxyz) — the only math utilities
  ik.py           # differential IK: pose error → joint velocities → qpos
  mapping.py      # VR→robot frame conversion + clutch (relative mapping)
  sources/        # TargetSource implementations: trajectory, mouse, pico (WebXR)
  real_robot.py   # RealRobot interface + DryRunRobot (default) + robot factory
  booster.py      # real Booster driver (K1 + T1): robot stands/balances itself (UpperBodyCustomControl), we command only the arms
  run_sim.py      # entry point: test-in-sim
  run_real.py     # entry point: test-in-real
  tools/          # setup_assets (download robot models), check_lesson
```

## Commands

```bash
uv sync                       # install everything (incl. dev group)
uv run setup-assets           # download the robot models (Booster K1 + T1, booster_assets)
uv run setup-assets --robot t1      # ...or just one robot
uv run test-in-sim            # demo: scripted trajectory in the MuJoCo viewer
uv run test-in-sim --robot-spec t1  # same, on the Booster T1 (default is k1)
uv run test-in-sim --source mouse   # drag the red/green target cubes yourself
uv run test-in-sim --source pico    # VR: open the printed URL in the Pico browser
uv run test-in-sim --headless       # no GUI; prints tracking error (CI/servers)
uv run test-in-real           # stream IK joint targets to a robot (default: dry-run)
uv sync --extra real          # add Booster's SDK (hardware only; Linux wheels)
uv run test-in-real --robot k1 --stand   # real K1: stand up, balance, then VR-drive the arms
uv run test-in-real --robot t1 --stand   # same, real T1 (the IK brain follows --robot)
uv run pytest                 # unit tests
uv run check-lesson           # enforce the Lesson Rule
uv run render-videos          # demo MP4s into video/ (offscreen, EGL; no display)
uv run render-videos --robot-spec t1     # same clips, rendered on the T1
```

## Robot model policy

- The robots are the **Booster K1** (default) and **Booster T1**, using
  Booster Robotics' official MuJoCo models from
  `BoosterRobotics/booster_assets` (BSD-3-Clause) — sim and real are the
  same machine. They are downloaded by `uv run setup-assets` into
  `assets/booster_k1/` and `assets/booster_t1/` and are **never committed**
  (see .gitignore).
- Vendor names are copied **verbatim**, including the `aa` prefixes — which
  sit on different joints per robot (K1: `aaleft_shoulder_pitch_joint`;
  T1: plain `left_shoulder_pitch_joint` but `aahead_yaw_joint`); never
  "normalize" them. Neither model ships a keyframe — each home pose is
  authored in its `RobotSpec`.
- Everything robot-specific lives in **one `RobotSpec` object per robot** in
  `src/teleop/robot.py` (XML paths, booster_assets folder, arm joint names,
  end-effector bodies, home pose, whether orientation is tracked). To use
  another robot: add a `RobotSpec` and pass `--robot-spec`. No other file
  may hard-code robot names (the driver's joint→motor tables in
  `booster.py` are keyed by `RobotSpec` names and checked against them in
  tests).

## Real-robot safety invariants (never remove these)

`run_real.py` must always keep: joint-limit clamping, per-tick joint velocity
clamping, a deadman (targets only move while the grip button is held), and a
watchdog (stale VR data ⇒ hold position). It must also keep its argparse
gates: deadman-free sources (anything but `pico`) are allowed only with
`--robot dryrun`, `--rate` values in the IK's unstable zone (gain·dt ≥ 2)
are rejected, `--stand` (a physical stand-up) requires a hardware robot
(`--robot k1` or `--robot t1`), and the IK brain must be the same robot as
the body (`--robot-spec` defaults to `--robot` and a mismatch is refused —
a K1 brain emits joint names a T1 driver would silently ignore). A
print-only robot (`DryRunRobot`) is the default so `test-in-real` is always
runnable without hardware. The Booster driver (`booster.py`, K1 + T1) adds
its own invariants: it re-clamps targets to joint limits (never trust the
caller), soft-starts by blending from the measured arm pose, commands only
the 8 arm motors (zero gains elsewhere ⇒ the onboard controller keeps
balancing — on the T1 that includes the waist), and on `disconnect()` hands
the arms back via `UpperBodyCustomControl(False)` — never damping mode,
which would drop a standing robot.

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
