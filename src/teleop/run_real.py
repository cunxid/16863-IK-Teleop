"""Entry point for ``uv run test-in-real`` — same brain, real body.

The trick: the "real" mode still runs MuJoCo! The simulated robot acts as the
*kinematic brain* — sources move the mocap targets, IK moves the simulated
joints — and every tick we read those joint angles and stream them to the
real robot's position controller. Sim-vs-real becomes one extra line.

Safety layers, in order (never remove any — see AGENTS.md):

    1. deadman   — targets only move while the grip button is held (the clutch)
    2. watchdog  — stale/absent VR data ⇒ sources return nothing ⇒ targets freeze
    3. IK clamps — joint limits and per-joint velocity are enforced in ik.step
    4. the robot — implementations must re-check limits on their side too

Default robot is ``dryrun`` (prints instead of moving), so this command is
always safe to try.
"""

from __future__ import annotations

import argparse
import time

from teleop import ik
from teleop.real_robot import ROBOT_NAMES, make_robot
from teleop.robot import SPECS, Robot
from teleop.sources import SOURCE_NAMES, make_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream IK joint targets to a real robot.")
    parser.add_argument("--robot", choices=ROBOT_NAMES, default="dryrun",
                        help="hardware backend (default: dry-run printer)")
    parser.add_argument("--source", choices=SOURCE_NAMES, default="pico",
                        help="where targets come from (default: VR controllers)")
    parser.add_argument("--robot-spec", choices=sorted(SPECS), default="g1",
                        help="kinematic model used as the IK brain")
    parser.add_argument("--rate", type=float, default=50.0, help="command rate [Hz]")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop after this many seconds (0 = run until Ctrl+C)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="pico mode: meters of hand motion per meter of controller motion")
    parser.add_argument("--port", type=int, default=8443, help="pico mode: HTTPS/WSS port")
    args = parser.parse_args()

    brain = Robot(SPECS[args.robot_spec])
    brain.reset()

    source = make_source(args.source, brain.targets(), scale=args.scale, port=args.port)
    real = make_robot(args.robot)

    dt = 1.0 / args.rate
    source.start()
    real.connect()
    print(f"streaming at {args.rate:.0f} Hz — Ctrl+C to stop")
    t0 = time.monotonic()
    try:
        while True:
            tick_start = time.monotonic()
            t = tick_start - t0
            if args.duration and t > args.duration:
                break

            targets = source.get_targets(t)
            if targets is not None:  # None ⇒ watchdog/deadman: hold position
                brain.set_targets(targets)
            ik.step(brain, dt)  # sim joints chase the targets (with clamps)
            real.send_joint_targets(brain.joint_targets())  # ...and the robot copies them

            leftover = dt - (time.monotonic() - tick_start)
            if leftover > 0:
                time.sleep(leftover)
    except KeyboardInterrupt:
        print("\nstopping (Ctrl+C)")
    finally:
        real.disconnect()
        source.stop()


if __name__ == "__main__":
    main()
