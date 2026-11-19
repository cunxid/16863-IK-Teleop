"""Target sources: everything that can say "the hands should go THERE".

A source produces target poses; it knows nothing about IK or the robot's
joints. This is the project's main interface — all three modes of the demo
are just different implementations of ``TargetSource``:

* ``trajectory`` — a scripted figure-8 (``trajectory.py``). Zero hardware.
* ``mouse``     — the user drags the target cubes in the MuJoCo viewer
                  (``MouseSource`` below — it does literally nothing!).
* ``pico``      — PICO VR controllers via a WebXR page (``pico.py``).

The contract: ``get_targets(t)`` returns a ``TargetFrame`` with the new
absolute target pose per hand, or ``None`` meaning "I have nothing to say,
leave the targets where they are". The main loops write whatever comes back
into the MuJoCo mocap bodies; the IK only ever looks at the mocap bodies.
"""

from __future__ import annotations

from dataclasses import dataclass

from teleop.mapping import Pose

SOURCE_NAMES = ("trajectory", "mouse", "pico")


@dataclass
class TargetFrame:
    """Desired pose per hand. A ``None`` hand means "don't move that one"."""

    left: Pose | None = None
    right: Pose | None = None


class TargetSource:
    """Interface. Subclasses override any of these three methods."""

    def start(self) -> None:
        """Called once before the loop (e.g. start a server thread)."""

    def get_targets(self, t: float) -> TargetFrame | None:
        """Called every frame with elapsed time t [s]."""
        return None

    def stop(self) -> None:
        """Called once after the loop (e.g. shut the server down)."""


class MouseSource(TargetSource):
    """The "do nothing" source, and that's the point.

    In mouse mode the *viewer* moves the targets: double-click a target cube,
    then Ctrl+drag (Ctrl+right-drag to translate). The viewer writes directly
    into ``data.mocap_pos``, the IK reads the mocap bodies, so there is
    nothing left for this class to do. One shared target convention = free
    interactivity.
    """


def make_source(name: str, initial: TargetFrame, *, scale: float = 1.0, port: int = 8443) -> TargetSource:
    """Build a source by name. ``initial`` = where the hands start (so every
    source begins exactly at the current hand poses and nothing jumps)."""
    if name == "trajectory":
        from teleop.sources.trajectory import TrajectorySource

        return TrajectorySource(initial)
    if name == "mouse":
        return MouseSource()
    if name == "pico":
        from teleop.sources.pico import PicoSource  # imported lazily: needs network deps

        return PicoSource(initial, scale=scale, port=port)
    raise ValueError(f"unknown source {name!r}; pick one of {SOURCE_NAMES}")
