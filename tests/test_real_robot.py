"""Tests for the real-robot boundary — no hardware, no SDK import needed.

The driver's per-tick math (``arm_setpoints``) is a pure function exactly so
it can be tested here; everything that needs the Booster SDK stays behind
``connect()`` and is exercised only on the real robot. Everything is checked
for both supported robots (K1 and T1).
"""

import pytest

from teleop.booster import JOINT_TO_MOTOR, MOTOR_COUNT, BoosterRobot, arm_setpoints
from teleop.real_robot import DryRunRobot, make_robot
from teleop.robot import SPECS, Robot

ROBOTS = sorted(JOINT_TO_MOTOR)  # ["k1", "t1"]


@pytest.mark.parametrize("name", ROBOTS)
def test_joint_to_motor_covers_exactly_the_arm_joints(name):
    spec_joints = [n for arm in SPECS[name].arms for n in arm.joint_names]
    assert sorted(JOINT_TO_MOTOR[name]) == sorted(spec_joints)
    # Arms are motors 2-9 on both robots (JointIndexK1 / JointIndex: head is
    # 0-1; what follows the arms differs — K1 legs, T1 waist then legs).
    assert sorted(JOINT_TO_MOTOR[name].values()) == list(range(2, 10))


def test_motor_counts_match_the_sdk_layouts():
    assert MOTOR_COUNT == {"k1": 22, "t1": 23}  # kJointCntK1 / kJointCnt


@pytest.mark.parametrize("name", ROBOTS)
def test_make_robot_returns_the_right_backend_without_the_sdk(name):
    assert isinstance(make_robot("dryrun"), DryRunRobot)
    # Building the hardware driver must not import the SDK (that happens in
    # connect) and must pick that robot's joint table.
    driver = make_robot(name, stand=True, iface="eth0")
    assert isinstance(driver, BoosterRobot)
    assert driver.joint_to_motor == JOINT_TO_MOTOR[name]
    assert driver.motor_count == MOTOR_COUNT[name]


@pytest.mark.parametrize("name", ROBOTS)
def test_arm_setpoints_blends_from_measured_start_to_target(name):
    table = JOINT_TO_MOTOR[name]
    joint = "left_elbow_yaw_joint"  # spelled the same on both robots
    motor = table[joint]
    targets, start = {joint: 1.0}, {joint: 0.0}
    assert arm_setpoints(targets, start, 0.0, table) == {motor: 0.0}
    assert arm_setpoints(targets, start, 1.0, table) == {motor: 1.0}
    assert abs(arm_setpoints(targets, start, 0.5, table)[motor] - 0.5) < 1e-12


def test_arm_setpoints_reclamps_to_joint_limits():
    table = JOINT_TO_MOTOR["k1"]
    joint = "left_elbow_yaw_joint"
    limits = {joint: (-1.0, 1.0)}
    out = arm_setpoints({joint: 99.0}, {joint: 1.0}, 1.0, table, limits=limits)
    assert out == {table[joint]: 1.0}


def test_arm_setpoints_skips_joints_that_are_not_arm_motors():
    # A K1 brain's 'aa'-prefixed shoulder name means nothing to the T1 table
    # (and vice versa) — it must be ignored, not misrouted to a wrong motor.
    assert arm_setpoints({"not_a_joint": 1.0}, {}, 1.0, JOINT_TO_MOTOR["k1"]) == {}
    assert arm_setpoints({"aaleft_shoulder_pitch_joint": 1.0}, {}, 1.0,
                         JOINT_TO_MOTOR["t1"]) == {}


@pytest.mark.parametrize("name", ROBOTS)
def test_robot_joint_limits_cover_all_arm_joints(name):
    robot = Robot(SPECS[name])
    limits = robot.joint_limits()
    assert sorted(limits) == sorted(JOINT_TO_MOTOR[name])
    for lo, hi in limits.values():
        assert lo < hi
