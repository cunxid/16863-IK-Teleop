"""The real-robot boundary: everything hardware hides behind one tiny interface.

``run_real.py`` computes joint targets with the exact same IK as the sim —
the robot only ever receives *joint positions by name*. That makes the
hardware interface three methods, and makes "sim vs real" a one-line swap.

Implementations:

* ``DryRunRobot`` — the default. No hardware: it just prints what *would*
  be sent, at a readable rate. `uv run test-in-real` therefore always works.
* ``FaunaRobot``  — the slot for the real Fauna robot. The methods raise
  ``NotImplementedError`` with instructions; filling them in with the
  vendor SDK is lesson §9's exercise. Nothing else in the codebase changes.
"""

from __future__ import annotations

import time


class RealRobot:
    """Interface. Position-controlled joints, addressed by MuJoCo joint name."""

    def connect(self) -> None:
        """Open the connection and put the robot in a safe, compliant mode."""

    def send_joint_targets(self, targets: dict[str, float]) -> None:
        """Command joint positions [rad], keyed by joint name.

        Called at a fixed rate (default 50 Hz) with values that are already
        joint-limit-clamped and velocity-limited by the caller. Implementations
        must still enforce their own hardware limits — never trust the caller.
        """
        raise NotImplementedError

    def disconnect(self) -> None:
        """Stop motion and release the connection. Must be safe to call twice."""


class DryRunRobot(RealRobot):
    """Prints commands instead of moving motors. The default for test-in-real."""

    def __init__(self, print_period: float = 1.0):
        self.print_period = print_period
        self._last_print = 0.0
        self._n_sent = 0

    def connect(self) -> None:
        print("[dry-run] connected (no hardware; commands will be printed)")

    def send_joint_targets(self, targets: dict[str, float]) -> None:
        self._n_sent += 1
        now = time.monotonic()
        if now - self._last_print < self.print_period:
            return
        self._last_print = now
        sample = ", ".join(f"{name}={value:+.3f}" for name, value in list(targets.items())[:4])
        print(f"[dry-run] #{self._n_sent:06d} {len(targets)} joints | {sample}, ...")

    def disconnect(self) -> None:
        print(f"[dry-run] disconnected after {self._n_sent} commands")


class FaunaRobot(RealRobot):
    """Where the real Fauna robot plugs in (see lesson §9).

    To bring the real robot up, implement the three methods with the vendor
    SDK and map MuJoCo joint names → motor ids in ``JOINT_TO_MOTOR``.
    Keep the safety order: connect in compliant/low-gain mode, ramp gains up,
    and make disconnect() stop motion even if called mid-stream.
    """

    JOINT_TO_MOTOR: dict[str, int] = {
        # "left_shoulder_pitch_joint": 0,   # ← fill in from the Fauna SDK docs
    }

    def connect(self) -> None:
        raise NotImplementedError(
            "FaunaRobot is a teaching stub. Implement connect/send/disconnect "
            "with the Fauna SDK (lesson §9), or run with --robot dryrun."
        )

    def send_joint_targets(self, targets: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        pass  # a stub never moved anything, so there is nothing to stop


ROBOT_NAMES = ("dryrun", "fauna")


def make_robot(name: str) -> RealRobot:
    if name == "dryrun":
        return DryRunRobot()
    if name == "fauna":
        return FaunaRobot()
    raise ValueError(f"unknown robot {name!r}; pick one of {ROBOT_NAMES}")
