"""Render demo videos offscreen (``uv run render-videos``) — no display needed.

Three clips land in ``video/``, named after the robot (``--robot-spec``,
default k1 — e.g. ``k1_01_figure8.mp4``):

* ``<robot>_01_figure8.mp4`` — the default demo: lift-in, then both hands
  trace a figure-8 (exactly what ``test-in-sim`` shows in the viewer).
* ``<robot>_02_pico_clutch.mp4`` — a scripted "fake headset" drives the real
  Pico pipeline (``PicoSource.ingest`` → frame conversion → clutch → IK). The
  small sphere is the controller: bright while the grip is held, dim when
  released. Watch the release phase: the sphere flies away, the robot does not
  move — and after the re-grab, motion resumes with no jump. That is the clutch.
* ``<robot>_03_workspace_edge.mp4`` — the targets sweep far beyond the arms'
  reach and come back: damped least squares strains gracefully instead of
  exploding (lesson §5's damping at work).

Rendering uses MuJoCo's offscreen ``Renderer``. ``MUJOCO_GL=egl`` is set
automatically (an env var you set yourself wins; ``--gl osmesa`` if your
machine lacks EGL). The sim runs kinematically at 60 Hz; frames are written
at 30 fps.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

from teleop import REPO_ROOT

VIDEO_DIR = REPO_ROOT / "video"
SIM_HZ = 60.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the demo videos offscreen.")
    parser.add_argument("--gl", choices=("egl", "osmesa", "glfw"), default="egl",
                        help="offscreen GL backend (a preset MUJOCO_GL env var wins)")
    parser.add_argument("--robot-spec", default="k1",
                        help="which robot to render (a RobotSpec name from robot.py)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # Must happen before the first `import mujoco` anywhere in the process —
    # which is also why robot.py (it imports mujoco) is imported only below.
    os.environ.setdefault("MUJOCO_GL", args.gl)

    from teleop.robot import SPECS

    if args.robot_spec not in SPECS:
        parser.error(f"--robot-spec {args.robot_spec!r}: unknown robot; "
                     f"pick one of {', '.join(sorted(SPECS))}")

    VIDEO_DIR.mkdir(exist_ok=True)
    render_figure8(args)
    render_pico_clutch(args)
    render_workspace_edge(args)
    print(f"\nall clips written to {VIDEO_DIR}")


# ----------------------------------------------------------------- clips ---


def render_figure8(args: argparse.Namespace) -> None:
    from teleop.robot import SPECS, Robot
    from teleop.sources.trajectory import TrajectorySource

    robot = Robot(SPECS[args.robot_spec])
    robot.reset()
    source = TrajectorySource(robot.targets())

    def drive(t: float) -> None:
        robot.set_targets(source.get_targets(t))

    record(args, "01_figure8.mp4", robot, drive, duration=3.0 + 2 * source.period)


def render_pico_clutch(args: argparse.Namespace) -> None:
    from teleop.mapping import XR_TO_MJ_MAT
    from teleop.robot import SPECS, Robot
    from teleop.sources.pico import PicoSource

    robot = Robot(SPECS[args.robot_spec])
    robot.reset()
    # The server is never started: we feed messages straight into ingest(),
    # which is everything after the network — conversion, clutch, watchdog.
    source = PicoSource(robot.targets(), scale=1.0, port=9999)

    # Where to *display* the controller spheres (schematic, beside the robot,
    # at roughly chest height — which scales with the trunk height).
    anchor_z = robot.spec.home_base_height + 0.4
    display_anchor = {"left": np.array([0.0, 0.45, anchor_z]),
                      "right": np.array([0.0, -0.45, anchor_z])}
    latest_hands: dict[str, dict | None] = {"left": None, "right": None}

    def drive(t: float) -> None:
        msg = {"t": t, "left": left_script(t), "right": right_script(t)}
        latest_hands.update(left=msg["left"], right=msg["right"])
        source.ingest(msg)
        frame = source.get_targets(t)
        if frame is not None:
            robot.set_targets(frame)

    def decorate(scene, t: float) -> None:
        for name, hand in latest_hands.items():
            if hand is None:
                continue
            pos = display_anchor[name] + XR_TO_MJ_MAT @ np.array(hand["pos"])
            alpha = 0.9 if hand["grip"] else 0.25  # bright = gripping (deadman on)
            rgba = (0.2, 0.4, 1.0, alpha) if name == "left" else (1.0, 0.6, 0.1, alpha)
            add_marker(scene, pos, rgba)

    record(args, "02_pico_clutch.mp4", robot, drive, duration=11.0, decorate=decorate)


def left_script(t: float) -> dict | None:
    """The left controller's story: grab → move → release & fly away → re-grab."""
    forward = -0.13 * seg(t, 1.0, 2.6)  # XR forward is -z  ⇒ robot +x
    up = 0.10 * seg(t, 2.6, 3.8)  # XR up is +y      ⇒ robot +z
    inward = 0.06 * seg(t, 3.8, 4.8)  # XR +x is robot -y: slide toward center
    if t < 5.2:
        pos, grip = [inward, up, forward], t >= 1.0
    elif t < 7.0:  # released: the controller leaves, the robot MUST hold still
        pos, grip = [0.45, up, forward], False
    else:  # re-grab far away: no jump — only new deltas count
        pos, grip = [0.45, up, forward - 0.10 * seg(t, 7.0, 9.5)], True
    return hand_message(pos, 0.0, grip)


def right_script(t: float) -> dict | None:
    """The right controller simply stirs a small circle while gripped."""
    if t < 2.0:
        return hand_message([0.0, 0.0, 0.0], 0.0, grip=False)
    radius = 0.07 * seg(t, 2.0, 3.0)
    angle = 2.0 * math.pi * (t - 2.0) / 4.0
    return hand_message([radius * math.cos(angle), radius * math.sin(angle), 0.0], 0.0, grip=True)


def hand_message(pos_xr: list[float], yaw_about_up: float, grip: bool) -> dict:
    """Build one hand entry in the exact JSON shape the WebXR page sends."""
    half = 0.5 * yaw_about_up
    return {
        "pos": list(pos_xr),
        "quat_xyzw": [0.0, math.sin(half), 0.0, math.cos(half)],  # about XR +y
        "grip": grip,
        "trigger": 1.0 if grip else 0.0,
    }


def render_workspace_edge(args: argparse.Namespace) -> None:
    from teleop.robot import SPECS, Robot
    from teleop.sources.trajectory import TrajectorySource

    robot = Robot(SPECS[args.robot_spec])
    robot.reset()
    lift = TrajectorySource(robot.targets(), amplitude=0.0)  # lift-in only, no 8
    lifted = lift.get_targets(lift.ramp_time)  # where the hands settle
    direction = np.array([0.8, 0.0, 0.6])
    direction /= np.linalg.norm(direction)

    def drive(t: float) -> None:
        if t < lift.ramp_time:
            robot.set_targets(lift.get_targets(t))
            return
        stretch = 0.40 * math.sin(math.pi * (t - lift.ramp_time) / 9.0)  # out and back
        frame = lift.get_targets(lift.ramp_time)
        frame.left.pos = lifted.left.pos + stretch * direction
        frame.right.pos = lifted.right.pos + stretch * direction
        robot.set_targets(frame)

    record(args, "03_workspace_edge.mp4", robot, drive, duration=12.0)


# ------------------------------------------------------------- machinery ---


def record(args: argparse.Namespace, filename: str, robot, drive, duration: float,
           decorate=None) -> None:
    """Run the usual sim loop; write every (SIM_HZ/fps)-th frame to an MP4."""
    import imageio
    import mujoco

    from teleop import ik

    # Framing scales with the robot's trunk height (K1: 0.552 m → distance
    # 1.8 m; the taller T1 gets a wider shot automatically).
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation = 140.0, -20.0
    camera.distance = 3.3 * robot.spec.home_base_height
    camera.lookat[:] = [0.05, 0.0, robot.spec.home_base_height]

    path = VIDEO_DIR / f"{robot.spec.name}_{filename}"
    renderer = mujoco.Renderer(robot.model, height=args.height, width=args.width)
    writer = imageio.get_writer(str(path), fps=args.fps, quality=8)
    steps_per_frame = max(1, round(SIM_HZ / args.fps))
    dt = 1.0 / SIM_HZ
    try:
        for i in range(int(duration * SIM_HZ)):
            t = i * dt
            drive(t)  # a source (or script) moves the mocap targets
            ik.step(robot, dt)  # the arms chase them — same loop as run_sim
            if i % steps_per_frame == 0:
                renderer.update_scene(robot.data, camera)
                if decorate is not None:
                    decorate(renderer.scene, t)
                writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
    print(f"wrote {path.relative_to(REPO_ROOT)}  ({duration:.0f} s, {path.stat().st_size / 1e6:.1f} MB)")


def add_marker(scene, pos: np.ndarray, rgba: tuple[float, float, float, float]) -> None:
    """Append one sphere to the render scene (visualization only, no physics)."""
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.028, 0.0, 0.0]),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def seg(t: float, t0: float, t1: float) -> float:
    """Smoothstep from 0 to 1 as t goes t0 → t1 (0 before, 1 after)."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    x = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


if __name__ == "__main__":
    main()
