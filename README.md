# ik-teleop — VR teleoperation of a humanoid's upper body (a teaching project)

Move a PICO VR controller; the robot's hand follows — via differential inverse
kinematics (IK), with MuJoCo as the backend.

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
uv run setup-assets # 2. download the robot model (Unitree G1, from mujoco_menagerie)
uv run test-in-sim  # 3. a MuJoCo window opens: both hands draw a figure-8 ✨
```

## test-in-sim (three ways to play)

```bash
uv run test-in-sim                   # scripted trajectory: hands draw an 8; proves the whole pipeline
uv run test-in-sim --source mouse    # mouse: double-click a red/green target cube, Ctrl+drag to move it
uv run test-in-sim --source pico     # VR: the terminal prints a URL ↓
uv run test-in-sim --headless        # no window; prints tracking error (servers/CI)
```

**Pico mode**: put the headset and the PC on the same Wi-Fi. In the Pico
browser, open the `https://<PC-IP>:8443` URL printed in the terminal, accept
the self-signed-certificate warning once, and tap **Enter VR**.
**Hold the grip button (side of the controller)** to make that hand follow
you; release and it freezes (a clutch design — prevents accidents and lets you
re-position your arm comfortably between grabs).

## test-in-real (driving a real robot)

```bash
uv run test-in-real                      # default dry-run: prints the joint commands it would send
uv run test-in-real --source trajectory  # rehearse the real-robot pipeline without a headset
uv run test-in-real --robot fauna        # the real Fauna robot (implement the SDK hookup per lesson §9)
```

The real-robot loop always keeps four safety layers: joint limits, per-joint
speed caps, the grip deadman switch, and a stale-VR-data watchdog (see
`src/teleop/run_real.py`).

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
| `src/teleop/real_robot.py` | hardware interface (DryRun / Fauna stub) | §9 |

## Development

```bash
uv run pytest        # unit tests (pure math + IK convergence; all windowless)
uv run check-lesson  # verify every source file has a lesson walkthrough (this repo's hard rule)
```

Contributor rules live in [AGENTS.md](AGENTS.md) — the most important one:
**every code change must update the lesson in the same change.**

## FAQ

- **Which robot?** The target machine is Fauna Robotics' **Sprout** (6-DoF
  arms + 1-DoF grippers). Its official URDF/MJCF ships only with the
  login-gated Fauna SDK, so the demo uses the Unitree G1 (mujoco_menagerie,
  BSD-3, 7-DoF arms) — another dual-arm humanoid — as a stand-in. Everything
  robot-specific lives in one `RobotSpec` in `robot.py`; once you have the
  Sprout model, add a spec and switch.
- **Browser says "not secure"?** WebXR requires HTTPS; on a LAN we use a
  self-signed certificate — tap "Advanced → Proceed". The certificate is
  generated into `.certs/` on first run.
- **No window on Linux?** The viewer needs a display/desktop environment; on a
  server use `--headless`.
