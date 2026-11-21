# ik-teleop — VR teleoperation of a humanoid's upper body (a teaching project)

Move a PICO VR controller; a **Booster humanoid**'s hand follows — via
differential inverse kinematics (IK), with MuJoCo as the backend. Two robots
are supported out of the box: the **K1** (0.95 m, the default) and the
**T1** (1.2 m) — pick one with `--robot-spec k1|t1`.

This is a **teaching project**: the code is written for students with no
robotics background, and every file maps to a section of the lesson,
[`lesson/index.html`](lesson/index.html) (open it in a browser and read along).

```
target source (trajectory / mouse / Pico VR) ──▶ mocap target cubes ──▶ differential IK ──▶ joint angles ──▶ sim / real robot
```

## Quickstart (3 commands)

Requires [uv](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync             # 1. install everything (creates the virtualenv automatically)
uv run setup-assets # 2. download the robot models (Booster K1 + T1, official booster_assets)
uv run test-in-sim  # 3. a MuJoCo window opens: both hands draw a figure-8 ✨
```

## test-in-sim (three ways to play)

```bash
uv run test-in-sim                   # scripted trajectory: hands draw an 8; proves the whole pipeline
uv run test-in-sim --robot-spec t1   # the same demo on the Booster T1 (default: k1)
uv run test-in-sim --source mouse    # mouse: double-click a red/green target cube, Ctrl+drag to move it
uv run test-in-sim --source pico     # VR: the terminal prints a URL ↓
uv run test-in-sim --headless        # no window; prints tracking error (servers/CI)
```

**Pico mode**: put the headset and the PC on the same Wi-Fi. In the Pico
browser, open the `https://<PC-IP>:8443` URL printed in the terminal, accept
the self-signed-certificate warning once, and tap **Enter VR**.
**Hold the grip button (side of the controller)** to make that hand follow
you; release and it freezes (a clutch design — prevents accidents and lets you
re-position your arm comfortably between grabs). If motions feel rotated,
face the robot's forward direction and **press A (or X)** — "forward" is
re-aligned to your gaze (the view flashes blue). Only controller *position*
drives the robot; wrist rotation is ignored (4-DoF arms track position only).

## test-in-real (driving a real robot)

```bash
uv run test-in-real                      # default dry-run: prints the joint commands it would send
uv run test-in-real --source trajectory  # rehearse the real-robot pipeline without a headset
uv sync --extra real                     # install Booster's SDK (hardware-only extra, Linux)
uv run test-in-real --robot k1 --stand   # the real K1: stand up, balance, then VR-drive the arms
uv run test-in-real --robot t1 --stand   # the same on a real T1 (the IK brain follows --robot)
```

With `--robot k1` or `--robot t1` the robot keeps balancing itself — the
driver enables `UpperBodyCustomControl` and commands only the 8 arm motors;
`--stand` first runs the stand-up sequence (prepare pose → balance
controller). See lesson §9.

The real-robot loop always keeps four safety layers: joint limits, per-joint
speed caps, the grip deadman switch, and a stale-VR-data watchdog (see
`src/teleop/run_real.py`). Two gates are enforced up front: a robot other
than `dryrun` requires `--source pico` (the only source with a deadman), and
`--rate` values low enough to destabilize the IK integration are rejected.

## Code map

| File | Job | Lesson |
| --- | --- | --- |
| `src/teleop/robot.py` | load the model, joint/end-effector indices, mocap target read/write | §2–§3 |
| `src/teleop/quat.py` | quaternion helpers (wxyz convention) | §3 |
| `src/teleop/ik.py` | differential IK (damped least squares) | §5 |
| `src/teleop/sources/` | target-source interface: trajectory / mouse / pico | §4, §7, §8 |
| `src/teleop/mapping.py` | VR→robot frame conversion + the clutch | §8 |
| `src/teleop/run_sim.py` | `test-in-sim` entry point | §6 |
| `src/teleop/run_real.py` | `test-in-real` entry point | §9 |
| `src/teleop/real_robot.py` | hardware interface (DryRun default) | §9 |
| `src/teleop/booster.py` | real Booster driver (K1 + T1): robot balances, we drive the arms | §9 |

## Development

```bash
uv run pytest        # unit tests (pure math + IK convergence; all windowless)
uv run check-lesson  # verify every source file has a lesson walkthrough (this repo's hard rule)
uv run render-videos # write demo MP4s into video/ (offscreen — works without a display)
```

Contributor rules live in [AGENTS.md](AGENTS.md) — the most important one:
**every code change must update the lesson in the same change.**

## FAQ

- **Which robot?** The **Booster K1** (0.95 m, 22 DoF, the default) or the
  **Booster T1** (1.2 m, 23 DoF — the K1 layout plus a waist joint), both
  using Booster Robotics' official MuJoCo models from
  `BoosterRobotics/booster_assets` (BSD-3) — the sim drives the same robot
  you deploy on. Both have 4-DoF arms (no wrists), so the IK tracks hand
  *position* only (lesson §5 explains the DoF budget). Everything
  robot-specific lives in one `RobotSpec` per robot in `robot.py`; a
  different robot is one new spec.
- **Browser says "not secure"?** WebXR requires HTTPS; on a LAN we use a
  self-signed certificate — tap "Advanced → Proceed". The certificate is
  generated into `.certs/` on first run.
- **No window on Linux?** The viewer needs a display/desktop environment; on a
  server use `--headless`.
- **macOS?** MuJoCo's interactive viewer must run under Apple's `mjpython`
  launcher: `uv run mjpython -m teleop.run_sim`. (`--headless` and all tests
  work under plain Python.)
