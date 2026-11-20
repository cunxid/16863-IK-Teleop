"""Tests for the real-robot boundary — no hardware, no SDK import needed.

The driver's per-tick math (``arm_setpoints``) is a pure function exactly so
it can be tested here; everything that needs the Booster SDK stays behind
``connect()`` and is exercised only on the real robot.
"""

from teleop.booster_k1 import JOINT_TO_MOTOR, BoosterK1Robot, arm_setpoints
from teleop.real_robot import DryRunRobot, make_robot
from teleop.robot import SPECS, Robot


def test_joint_to_motor_covers_exactly_the_k1_arm_joints():
    spec_joints = [n for arm in SPECS["k1"].arms for n in arm.joint_names]
    assert sorted(JOINT_TO_MOTOR) == sorted(spec_joints)
    # JointIndexK1: head = 0-1, arms = 2-9 (legs 10-21 are never commanded).
    assert sorted(JOINT_TO_MOTOR.values()) == list(range(2, 10))


def test_make_robot_returns_the_right_backend_without_the_sdk():
    assert isinstance(make_robot("dryrun"), DryRunRobot)
    # Building the k1 driver must not import the SDK (that happens in connect).
    assert isinstance(make_robot("k1", stand=True, iface="eth0"), BoosterK1Robot)


def test_arm_setpoints_blends_from_measured_start_to_target():
    name = "left_elbow_yaw_joint"
    motor = JOINT_TO_MOTOR[name]
    targets, start = {name: 1.0}, {name: 0.0}
    assert arm_setpoints(targets, start, blend=0.0) == {motor: 0.0}
    assert arm_setpoints(targets, start, blend=1.0) == {motor: 1.0}
    assert abs(arm_setpoints(targets, start, blend=0.5)[motor] - 0.5) < 1e-12


def test_arm_setpoints_reclamps_to_joint_limits():
    name = "left_elbow_yaw_joint"
    motor = JOINT_TO_MOTOR[name]
    limits = {name: (-1.0, 1.0)}
    out = arm_setpoints({name: 99.0}, {name: 1.0}, blend=1.0, limits=limits)
    assert out == {motor: 1.0}


def test_arm_setpoints_skips_joints_that_are_not_arm_motors():
    assert arm_setpoints({"not_a_joint": 1.0}, {}, blend=1.0) == {}


def test_robot_joint_limits_cover_all_arm_joints():
    robot = Robot(SPECS["k1"])
    limits = robot.joint_limits()
    assert sorted(limits) == sorted(JOINT_TO_MOTOR)
    for lo, hi in limits.values():
        assert lo < hi
