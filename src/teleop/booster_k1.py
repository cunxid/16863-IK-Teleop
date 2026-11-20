"""The real K1 driver: the last stop of the pipeline, joint targets → motors.

``run_real.py`` hands this file a dict of arm joint angles 50 times a second.
This file turns each dict into one Booster-SDK ``LowCmd`` message. The design
in one sentence: **the robot keeps balancing itself; we only borrow the arms.**

That split is a built-in robot feature, ``UpperBodyCustomControl``:

* the onboard controller keeps owning the legs (and anything else we don't
  command) — the robot stands, balances, catches pushes;
* our ``LowCmd`` position targets drive just the 8 arm motors.

So unlike full low-level mode (``RobotMode.kCustom``), balance never becomes
our problem, and letting go (``disconnect``) simply hands the arms back to the
robot — which stays standing.

Three safety details to notice (they mirror lesson §9's layer table):

* targets are re-clamped to joint limits here, even though the IK already
  clamped them — a driver never trusts its caller (layer 4);
* the first ``BLEND_SECONDS`` blend from the *measured* arm pose to the
  commanded one, so the arms can never jump at takeover;
* non-arm motors get kp = kd = 0: a zero-gain command is "not a command",
  which is exactly what leaves them to the onboard balance controller.

The SDK import lives inside ``connect()`` so everything else (tests, dry-run)
works without the hardware extra; install it with ``uv sync --extra real``.
"""

from __future__ import annotations

import time

from teleop.real_robot import RealRobot

# Our MuJoCo joint names (vendor-verbatim, from RobotSpec) → the SDK's
# JointIndexK1 motor numbering: head = 0–1, arms = 2–9, legs = 10–21.
JOINT_TO_MOTOR: dict[str, int] = {
    "aaleft_shoulder_pitch_joint": 2,
    "left_shoulder_roll_joint": 3,
    "left_elbow_pitch_joint": 4,
    "left_elbow_yaw_joint": 5,
    "aaright_shoulder_pitch_joint": 6,
    "right_shoulder_roll_joint": 7,
    "right_elbow_pitch_joint": 8,
    "right_elbow_yaw_joint": 9,
}
K1_MOTOR_COUNT = 22  # the SDK's kJointCntK1: every LowCmd carries all 22

# Arm position-control gains. Booster's own arm examples use kp=60 with
# kd 1.5–3; we take the soft end. 0x0A is the SDK's position-control mode.
ARM_KP = 60.0
ARM_KD = 1.5
POSITION_MODE = 0x0A

BLEND_SECONDS = 2.0  # soft-start: measured pose → commanded pose
PREPARE_SECONDS = 8.0  # generous wait for the stand-up to finish
WALK_SETTLE_SECONDS = 2.0  # let the balance controller settle before takeover
LOW_STATE_TIMEOUT = 5.0  # how long to wait for the robot's first state message


def arm_setpoints(
    targets: dict[str, float],
    start: dict[str, float],
    blend: float,
    limits: dict[str, tuple[float, float]] | None = None,
) -> dict[int, float]:
    """One tick of pure math: clamp to limits, blend from the measured start.

    Returns ``{motor index: commanded angle [rad]}`` for the arm motors only.
    ``blend`` runs 0 → 1 over the first seconds: 0 holds the pose the arms
    were measured in at takeover, 1 follows ``targets`` exactly. A joint
    absent from ``targets`` is simply not commanded (the robot keeps it).
    """
    out: dict[int, float] = {}
    for name, motor in JOINT_TO_MOTOR.items():
        if name not in targets:
            continue
        q = targets[name]
        if limits and name in limits:
            lo, hi = limits[name]
            q = min(max(q, lo), hi)
        out[motor] = (1.0 - blend) * start.get(name, q) + blend * q
    return out


class BoosterK1Robot(RealRobot):
    """Stream arm position targets to a standing, self-balancing Booster K1.

    ``stand=True`` adds a stand-up sequence before takeover: prepare pose
    (``RobotMode.kPrepare``), then the balance controller
    (``RobotMode.kWalking``). With ``stand=False`` the robot must already be
    standing (e.g. brought up with Booster's remote).
    """

    def __init__(
        self,
        joint_limits: dict[str, tuple[float, float]] | None = None,
        stand: bool = False,
        iface: str = "",
    ):
        self.joint_limits = dict(joint_limits or {})
        self.stand = stand
        self.iface = iface  # network interface to the robot; "" lets DDS pick
        self._sdk = None  # the lazily imported booster_robotics_sdk_python
        self._client = None
        self._publisher = None
        self._q_start: dict[str, float] = {}  # measured arm pose at takeover
        self._t_first: float | None = None  # when the first command went out

    def connect(self) -> None:
        try:
            import booster_robotics_sdk_python as sdk
        except ImportError as exc:
            raise RuntimeError(
                "Booster SDK not installed — it is the optional hardware "
                "extra:  uv sync --extra real   (Linux only; see lesson §9)"
            ) from exc
        self._sdk = sdk
        sdk.ChannelFactory.Instance().Init(0, self.iface)  # Fast DDS over LAN
        self._client = sdk.B1LocoClient()
        self._client.Init()
        if self.stand:
            self._stand_up()
        self._read_start_pose()
        print("[k1] taking over the arms (legs keep balancing on their own)")
        self._client.UpperBodyCustomControl(True)
        self._publisher = sdk.B1LowCmdPublisher()
        self._publisher.InitChannel()

    def _stand_up(self) -> None:
        """Prepare pose, then the balance controller — then the robot stands."""
        mode = self._sdk.RobotMode
        print(f"[k1] standing up: prepare pose ({PREPARE_SECONDS:.0f} s)...")
        self._client.ChangeMode(mode.kPrepare)
        time.sleep(PREPARE_SECONDS)
        print("[k1] standing up: switching to the balance controller...")
        self._client.ChangeMode(mode.kWalking)
        time.sleep(WALK_SETTLE_SECONDS)
        print("[k1] standing and balancing")

    def _read_start_pose(self) -> None:
        """Grab one state message: the measured arm angles seed the soft-start."""
        measured: list[float] = []

        def on_state(msg) -> None:
            if not measured:  # only the first message matters
                measured.extend(m.q for m in msg.motor_state_parallel)

        subscriber = self._sdk.B1LowStateSubscriber(on_state)
        subscriber.InitChannel()
        deadline = time.monotonic() + LOW_STATE_TIMEOUT
        while not measured and time.monotonic() < deadline:
            time.sleep(0.02)
        subscriber.CloseChannel()
        if not measured:
            raise RuntimeError(
                f"no robot state after {LOW_STATE_TIMEOUT:.0f} s — is the K1 "
                f"powered on and on this network (try --iface)?"
            )
        self._q_start = {n: measured[m] for n, m in JOINT_TO_MOTOR.items()}

    def send_joint_targets(self, targets: dict[str, float]) -> None:
        now = time.monotonic()
        if self._t_first is None:
            self._t_first = now
        blend = min(1.0, (now - self._t_first) / BLEND_SECONDS)
        setpoints = arm_setpoints(targets, self._q_start, blend, self.joint_limits)

        command = self._sdk.LowCmd()
        # PARALLEL matches the motor_state_parallel we read; for the arm
        # motors the two encodings are identical anyway (they differ in legs).
        command.cmd_type = self._sdk.LowCmdType.PARALLEL
        motors = []
        for index in range(K1_MOTOR_COUNT):
            motor = self._sdk.MotorCmd()
            motor.mode = POSITION_MODE
            motor.dq = 0.0
            motor.tau = 0.0
            if index in setpoints:
                motor.q = setpoints[index]
                motor.kp = ARM_KP
                motor.kd = ARM_KD
            else:
                motor.q = 0.0  # zero gains ⇒ the onboard controller keeps it
                motor.kp = 0.0
                motor.kd = 0.0
            motors.append(motor)
        command.motor_cmd = motors
        self._publisher.Write(command)

    def disconnect(self) -> None:
        if self._publisher is not None:
            self._publisher.CloseChannel()
            self._publisher = None
        if self._client is not None:
            try:
                # Hand the arms back to the onboard controller. The robot
                # keeps standing — safer than damping, which would drop it.
                self._client.UpperBodyCustomControl(False)
                print("[k1] arms handed back; robot keeps standing")
            except Exception as exc:
                print(f"[k1] could not hand back cleanly: {exc}")
            self._client = None
