"""The real-robot boundary: everything hardware hides behind one tiny interface.

``run_real.py`` computes joint targets with the exact same IK as the sim —
the robot only ever receives *joint positions by name*. That makes the
hardware interface three methods, and makes "sim vs real" a one-line swap.

Implementations:

* ``DryRunRobot`` — the default. No hardware: it just prints what *would*
  be sent, at a readable rate. `uv run test-in-real` therefore always works.
* ``BoosterK1Robot`` — the real Booster K1 driver, in its own module
  ``booster_k1.py``: the robot stands and balances itself while we command
  only the arms. Nothing else in the codebase knows the difference.
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


ROBOT_NAMES = ("dryrun", "k1")


def make_robot(
    name: str,
    joint_limits: dict[str, tuple[float, float]] | None = None,
    stand: bool = False,
    iface: str = "",
) -> RealRobot:
    """Build a robot backend. The k1-only options are harmless for dryrun."""
    if name == "dryrun":
        return DryRunRobot()
    if name == "k1":
        # Imported here, not at the top: booster_k1 imports RealRobot from
        # this module, so a top-level import would be circular.
        from teleop.booster_k1 import BoosterK1Robot

        return BoosterK1Robot(joint_limits=joint_limits, stand=stand, iface=iface)
    raise ValueError(f"unknown robot {name!r}; pick one of {ROBOT_NAMES}")
