"""Entry point for ``uv run test-in-sim`` — the whole pipeline, in simulation.

The main loop is four lines long, on purpose. Per frame (60 Hz):

    1. ask the source where the hands should be        (source.get_targets)
    2. write that into the mocap target bodies         (robot.set_targets)
    3. move every arm one small IK step toward them    (ik.step)
    4. draw                                            (viewer.sync)

Nothing here is simulation-specific except step 4 — ``run_real.py`` is the
same loop with "draw" replaced by "send to robot".

Note we never call ``mj_step`` (physics): this is a *kinematic* puppet show.
The IK writes joint angles directly, so the robot can't fall over and there
are no motor gains to tune — exactly what you want while learning IK.
"""

from __future__ import annotations

import argparse
import sys
import time

import mujoco.viewer

from teleop import ik
from teleop.robot import SPECS, Robot
from teleop.sources import SOURCE_NAMES, TargetSource, make_source

FPS = 60.0
HEADLESS_PASS_THRESHOLD = 0.03  # [m] max final tracking error for exit code 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Teleop the simulated robot (viewer or headless).")
    parser.add_argument("--source", choices=SOURCE_NAMES, default="trajectory",
                        help="where targets come from (default: scripted figure-8)")
    parser.add_argument("--robot-spec", choices=sorted(SPECS), default="g1",
                        help="which robot model to load")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="pico mode: meters of hand motion per meter of controller motion")
    parser.add_argument("--port", type=int, default=8443, help="pico mode: HTTPS/WSS port")
    parser.add_argument("--headless", action="store_true",
                        help="no window; run --duration seconds and report tracking error")
    parser.add_argument("--duration", type=float, default=10.0, help="headless run length [s]")
    args = parser.parse_args()

    robot = Robot(SPECS[args.robot_spec])
    robot.reset()  # home pose; mocap targets snap onto the hands

    source = make_source(args.source, robot.targets(), scale=args.scale, port=args.port)
    source.start()
    try:
        if args.headless:
            run_headless(robot, source, args.duration)
        else:
            run_viewer(robot, source)
    finally:
        source.stop()


def run_viewer(robot: Robot, source: TargetSource) -> None:
    dt = 1.0 / FPS
    print("viewer: double-click a target cube to select it, then "
          "Ctrl+right-drag moves it, Ctrl+left-drag rotates it "
          "(add Shift for the horizontal plane). Close the window to quit.")
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        t0 = time.monotonic()
        last_report = 0.0
        while viewer.is_running():
            frame_start = time.monotonic()
            t = frame_start - t0

            targets = source.get_targets(t)  # 1. where should the hands be?
            if targets is not None:
                robot.set_targets(targets)  # 2. write into mocap bodies
            errors = ik.step(robot, dt)  # 3. arms chase the mocap bodies
            viewer.sync()  # 4. draw (also applies your mouse drags to mocap)

            if t - last_report > 2.0:
                last_report = t
                report = "  ".join(f"{name}: {err * 1000:5.1f} mm" for name, err in errors.items())
                print(f"t={t:6.1f} s  tracking error  {report}")
            sleep_until_next_frame(frame_start, dt)


def run_headless(robot: Robot, source: TargetSource, duration: float) -> None:
    """Same loop without a window — the end-to-end smoke test (CI, servers)."""
    dt = 1.0 / FPS
    errors: dict[str, float] = {}
    for i in range(int(duration * FPS)):
        t = i * dt
        targets = source.get_targets(t)
        if targets is not None:
            robot.set_targets(targets)
        errors = ik.step(robot, dt)
        if i % int(FPS) == 0:
            report = "  ".join(f"{name}: {err * 1000:5.1f} mm" for name, err in errors.items())
            print(f"t={t:5.1f} s  tracking error  {report}")

    worst = max(errors.values())
    if worst > HEADLESS_PASS_THRESHOLD:
        sys.exit(f"FAIL: final tracking error {worst * 1000:.1f} mm "
                 f"> {HEADLESS_PASS_THRESHOLD * 1000:.0f} mm")
    print(f"PASS: final tracking error {worst * 1000:.1f} mm")


def sleep_until_next_frame(frame_start: float, dt: float) -> None:
    """Hold the loop at a steady 60 Hz (sleep off whatever time is left)."""
    leftover = dt - (time.monotonic() - frame_start)
    if leftover > 0:
        time.sleep(leftover)


if __name__ == "__main__":
    main()
