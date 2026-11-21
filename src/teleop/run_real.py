"""Entry point for ``uv run test-in-real`` — same brain, real body.

The trick: the "real" mode still runs MuJoCo! The simulated robot acts as the
*kinematic brain* — sources move the mocap targets, IK moves the simulated
joints — and every tick we read those joint angles and stream them to the
real robot's position controller. Sim-vs-real becomes one extra line.

Safety layers, in order (never remove any — see AGENTS.md):

    0. gates     — argparse refuses deadman-free sources on real hardware and
                   command rates in the IK's unstable zone (gain·dt ≥ 2)
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

# "mouse" needs the sim viewer (that's what moves the targets) — sim-only.
REAL_SOURCES = tuple(name for name in SOURCE_NAMES if name != "mouse")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream IK joint targets to a real robot.")
    parser.add_argument("--robot", choices=ROBOT_NAMES, default="dryrun",
                        help="hardware backend (default: dry-run printer)")
    parser.add_argument("--source", choices=REAL_SOURCES, default="pico",
                        help="where targets come from (default: VR controllers; "
                             "'mouse' is sim-only — use test-in-sim for that)")
    parser.add_argument("--robot-spec", choices=sorted(SPECS), default=None,
                        help="kinematic model used as the IK brain (default: match "
                             "--robot, or k1 for dryrun)")
    parser.add_argument("--rate", type=float, default=50.0, help="command rate [Hz]")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop after this many seconds (0 = run until Ctrl+C)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="pico mode: meters of hand motion per meter of controller motion")
    parser.add_argument("--port", type=int, default=8443, help="pico mode: HTTPS/WSS port")
    parser.add_argument("--stand", action="store_true",
                        help="hardware mode: command the robot to stand up and balance "
                             "(prepare pose, then the walking controller) before "
                             "taking over the arms; omit if it is already standing")
    parser.add_argument("--iface", default="",
                        help="hardware mode: network interface to the robot "
                             "(default: let the SDK's DDS layer choose)")
    args = parser.parse_args()

    # The IK brain must be the same robot as the body it drives: a K1 brain
    # produces K1 joint names, which a T1 driver would not recognize. With no
    # explicit --robot-spec, simply follow --robot (dryrun accepts anything).
    if args.robot_spec is None:
        args.robot_spec = args.robot if args.robot in SPECS else "k1"
    if args.robot in SPECS and args.robot_spec != args.robot:
        parser.error(f"--robot-spec {args.robot_spec} is a different robot than "
                     f"--robot {args.robot}; the IK brain must match the body "
                     f"(omit --robot-spec to match automatically)")

    # Safety gates (AGENTS.md invariants), checked before anything moves:
    # 1. only the pico source has a deadman + watchdog, so it is the only
    #    source allowed to drive a robot that is not the dry-run printer;
    if args.robot != "dryrun" and args.source != "pico":
        parser.error(f"--source {args.source} has no deadman/watchdog; a real robot "
                     f"requires --source pico (rehearse other sources with --robot dryrun)")
    # 2. the IK integrates with explicit Euler, which is unstable when
    #    gain·dt ≥ 2 — a too-low rate would stream a violent oscillation.
    gain = max(ik.IKSettings().pos_gain, ik.IKSettings().rot_gain)
    if args.rate <= gain / 2.0:
        parser.error(f"--rate {args.rate:g} Hz is unstable with IK gain {gain:g}/s "
                     f"(explicit Euler needs gain/rate < 2, else the commands "
                     f"oscillate at full speed instead of tracking); "
                     f"use a rate above {gain / 2.0:.0f} Hz — the default is 50")
    # 3. --stand physically moves a robot, so it only makes sense on one.
    if args.stand and args.robot not in SPECS:
        parser.error("--stand commands a physical stand-up and requires a "
                     "hardware robot (--robot k1 or --robot t1)")

    brain = Robot(SPECS[args.robot_spec])
    brain.reset()

    source = make_source(args.source, brain.targets(), scale=args.scale, port=args.port)
    real = make_robot(args.robot, joint_limits=brain.joint_limits(),
                      stand=args.stand, iface=args.iface)

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
