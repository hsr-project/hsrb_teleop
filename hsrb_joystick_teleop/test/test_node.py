#!/usr/bin/env python
'''
Copyright (c) 2024 TOYOTA MOTOR CORPORATION
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may be used
  to endorse or promote products derived from this software without specific
  prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS
LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
'''
import math
import os
import threading
from typing import (
    List,
    Optional,
    Tuple,
)

from ament_index_python import get_package_share_directory
from geometry_msgs.msg import (
    Twist,
    TwistStamped,
)
from hsrb_joystick_teleop.joystick_control import (
    DrivePowerControl,
    JoystickControlManager,
)
from hsrb_joystick_teleop.joystick_control_node import main
import pytest
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import GetParameters
import rclpy
from rclpy.action import ActionServer
from rclpy.duration import Duration
from rclpy.node import (
    MsgType,
    Node,
)
from rclpy.parameter import Parameter
from sensor_msgs.msg import (
    JointState,
    Joy,
)
from std_msgs.msg import (
    Bool,
    Empty,
)
from tmc_control_msgs.action import GripperApplyEffort
from tmc_msgs.msg import JointVelocity
from tmc_voice_msgs.msg import Voice
from trajectory_msgs.msg import JointTrajectory


# Waiting time until you accept the next command
_PUB_WAIT = 0.6
_SUB_TIMEOUT = 3.0
_CONNECTION_TIMEOUT = 10.0
_RATE = 10.0
_BUTTON_NUM = 13
_AXES_NUM = 6


class Subscriber:
    """Class to test if a topic has been issued."""

    def __init__(self, node: Node, msg_type: MsgType, topic: str) -> None:
        self._node = node
        self._subscriber = self._node.create_subscription(
            msg_type, topic, self.callback, 1)
        self._rate = self._node.create_rate(_RATE, self._node.get_clock())
        self._is_sub = False

    def wait_for_message(self, timeout: float) -> Tuple[bool, Optional[MsgType]]:
        start = self._node.get_clock().now()
        while not self._is_sub and (self._node.get_clock().now() - start) < Duration(seconds=timeout):
            self._rate.sleep()
        if not self._is_sub:
            return False, None
        self._is_sub = False
        return True, self._value

    def wait_until_connection_established(self, timeout: float) -> bool:
        # TODO(ono) get_publisher_countが実装されたら戻す
        # rate = self._node.create_rate(_RATE)
        start = self._node.get_clock().now()
        # while (self._subscriber.get_publisher_count() == 0
        #        and (self._node.get_clock().now() - start) < Duration(seconds=timeout)):
        #     rate.sleep()
        return (self._node.get_clock().now() - start) < Duration(seconds=timeout)

    def callback(self, msg: MsgType):
        self._value = msg
        self._is_sub = True


class DummyActionServer(Node):
    def __init__(self, node_name: str, action_type, topic: str) -> None:
        super().__init__(node_name)

        self._action_server = ActionServer(
            self, action_type, topic, execute_callback=self.callback)
        self._dummy_pub = self.create_publisher(Empty, f"{topic}/dummy", 1)
        self._result = action_type.Result()

    def callback(self, goal_handle):
        msg = Empty()
        self._dummy_pub.publish(msg)
        return self._result


class JoystickControlTest:
    def __init__(self, node: Node) -> None:
        self._node = node
        self.joy_pub = self._node.create_publisher(Joy, "joy", 1)
        self.joint_states_pub = self._node.create_publisher(
            JointState, "joint_states", 1)

        self.command_velocity_sub = Subscriber(self._node, Twist, "command_velocity")
        self.command_drive_power_sub = Subscriber(self._node, Bool, "command_drive_power")
        self.gripper_control_sub = Subscriber(self._node, JointTrajectory, "gripper_controller/joint_trajectory")
        self.endeffector_sub = Subscriber(
            self._node, TwistStamped, "pseudo_endeffector_controller/command_velocity")
        self.endeffector_with_base_sub = Subscriber(
            self._node, TwistStamped, "pseudo_endeffector_controller/command_velocity_with_base")
        self.joint_velocity_sub = Subscriber(
            self._node, JointVelocity, "pseudo_velocity_controller/ref_joint_velocity")
        self.joint_reference_sub = Subscriber(self._node, JointState, "safe_pose_changer/joint_reference")
        self.talk_request_sub = Subscriber(self._node, Voice, "talk_request")

        self.gripper_control_apply_force_sub = Subscriber(
            self._node, Empty, "gripper_controller/apply_force/dummy")
        self.gripper_control_grasp_sub = Subscriber(
            self._node, Empty, "gripper_controller/grasp/dummy")


@pytest.fixture
def setup(mocker):
    rclpy.init()
    joy_config_path = os.path.join(
        get_package_share_directory("hsrb_joystick_teleop"),
        "config",
        "test_joystick_control_config.yaml")
    test_node = rclpy.create_node(
        "test_node",
        cli_args=["--ros-args", "--params-file", joy_config_path],
        allow_undeclared_parameters=True,
        automatically_declare_parameters_from_overrides=True)
    apply_force_server = DummyActionServer('apply_force_server', GripperApplyEffort, 'gripper_controller/apply_force')
    grasp_server = DummyActionServer('grasp_server', GripperApplyEffort, 'gripper_controller/grasp')
    mocker.patch("rclpy.init")

    exit_flag = False

    def spin_func(node):
        # SingleTheDedEXECUTOR may make an infinite loop
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        executor.add_node(test_node)
        executor.add_node(apply_force_server)
        executor.add_node(grasp_server)
        while rclpy.ok():
            executor.spin_once()
            if exit_flag:
                break

    spin_mock = mocker.patch("rclpy.spin")
    spin_mock.side_effect = spin_func

    main_thread = threading.Thread(target=main, args=(joy_config_path,))
    main_thread.start()

    yield test_node

    exit_flag = True
    main_thread.join()

    test_node.destroy_node()
    # rclpy.shutdown is called in main


def get_parameters(node: Node, names: List[str]):
    client = node.create_client(GetParameters, "joystick_control_node/get_parameters")
    while not client.wait_for_service(timeout_sec=1.0):
        pass

    def send_request(names):
        req = GetParameters.Request()
        req.names = names
        return client.call(req).values

    def get_value(param_value: ParameterValue):
        _type = Parameter.Type(value=param_value.type)
        if _type == Parameter.Type.BOOL:
            return param_value.bool_value
        elif _type == Parameter.Type.INTEGER:
            return param_value.integer_value
        elif _type == Parameter.Type.DOUBLE:
            return param_value.double_value
        elif _type == Parameter.Type.STRING:
            return param_value.string_value
        elif _type == Parameter.Type.BYTE_ARRAY:
            return param_value.byte_array_value
        elif _type == Parameter.Type.BOOL_ARRAY:
            return param_value.bool_array_value
        elif _type == Parameter.Type.INTEGER_ARRAY:
            return param_value.integer_array_value
        elif _type == Parameter.Type.DOUBLE_ARRAY:
            return param_value.double_array_value
        elif _type == Parameter.Type.STRING_ARRAY:
            return param_value.string_array_value

    param_values = send_request(names)
    params = []
    for param_value in param_values:
        params.append(get_value(param_value))
    return params


# Togens of bogie movement
def test_base_control(setup):
    test_node = setup
    joy_ctrl = JoystickControlTest(test_node)
    assert joy_ctrl.command_velocity_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

    # Get parameters
    params = get_parameters(
        test_node,
        ["controls.base_control.button",
         "controls.base_control.dead_zone",
         "controls.base_control.axis.x",
         "controls.base_control.axis.y",
         "controls.base_control.axis.yaw",
         "controls.base_control.scale.x",
         "controls.base_control.scale.y",
         "controls.base_control.scale.yaw"])
    params_fast = get_parameters(
        test_node,
        ["controls.base_control_fast.button",
         "controls.base_control_fast.dead_zone",
         "controls.base_control_fast.axis.x",
         "controls.base_control_fast.axis.y",
         "controls.base_control_fast.axis.yaw",
         "controls.base_control_fast.scale.x",
         "controls.base_control_fast.scale.y",
         "controls.base_control_fast.scale.yaw"])

    # Manual creation of joystick MSG
    msg = Joy()
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    axes = [0.5, 0.8, -1.0]
    msg.axes = [0.0] * _AXES_NUM
    for i, idx in enumerate(params[2:5]):
        msg.axes[idx] = axes[i]
    joy_ctrl.joy_pub.publish(msg)

    is_sub, twist = joy_ctrl.command_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert pytest.approx(params[5] * msg.axes[params[2]]) == twist.linear.x
    assert pytest.approx(params[6] * msg.axes[params[3]]) == twist.linear.y
    assert pytest.approx(params[7] * msg.axes[params[4]]) == twist.angular.z

    # During high -speed mode
    msg.buttons = [0] * _BUTTON_NUM
    for idx in params_fast[0]:
        msg.buttons[idx] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, twist = joy_ctrl.command_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert pytest.approx(params_fast[5] * msg.axes[params_fast[2]]) == twist.linear.x
    assert pytest.approx(params_fast[6] * msg.axes[params_fast[3]]) == twist.linear.y
    assert pytest.approx(params_fast[7] * msg.axes[params_fast[4]]) == twist.angular.z

    # When the bogie movement button is separated, set the speed to 0
    msg.buttons = [0] * _BUTTON_NUM
    joy_ctrl.joy_pub.publish(msg)

    is_sub, twist = joy_ctrl.command_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert 0 == twist.linear.y
    assert 0 == twist.linear.x
    assert 0 == twist.angular.z

    # Speed ​​0 (normal time) if it does not exceed the unusual zone
    msg.buttons[params[0]] = 1
    dead_zone = params[1]
    msg.axes = [0.0] * _AXES_NUM
    for idx in params[2:5]:
        msg.axes[idx] = dead_zone - 0.01
    joy_ctrl.joy_pub.publish(msg)

    is_sub, twist = joy_ctrl.command_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert 0 == twist.linear.y
    assert 0 == twist.linear.x
    assert 0 == twist.angular.z

    # Speed ​​0 (in high -speed mode)
    msg.buttons = [0] * _BUTTON_NUM
    for idx in params_fast[0]:
        msg.buttons[idx] = 1
    dead_zone = params_fast[1]
    msg.axes = [0.0] * _AXES_NUM
    for idx in params[2:5]:
        msg.axes[idx] = dead_zone - 0.01
    assert is_sub
    assert 0 == twist.linear.y
    assert 0 == twist.linear.x
    assert 0 == twist.angular.z


# Joint single axis operation test
def test_single_joint_control(setup):
    test_node = setup
    joy_ctrl = JoystickControlTest(test_node)
    assert joy_ctrl.joint_velocity_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

    # Get parameters
    joint_name = "arm_lift_joint"
    params = get_parameters(
        test_node,
        ["controls.lift_control.button",
         f"controls.lift_control.{joint_name}.axis",
         f"controls.lift_control.{joint_name}.velocity"])
    params_fast = get_parameters(
        test_node,
        ["controls.lift_control_fast.button",
         f"controls.lift_control_fast.{joint_name}.axis",
         f"controls.lift_control_fast.{joint_name}.velocity"])

    # Manual creation of joystick MSG
    msg = Joy()
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    msg.axes = [0.0] * _AXES_NUM
    msg.axes[params[1]] = 0.5
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_vel = joy_ctrl.joint_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert len(joint_vel.name) == 1
    assert len(joint_vel.velocity) == 1
    assert joint_vel.name == [joint_name]
    assert pytest.approx(params[2] * msg.axes[params[1]]) == joint_vel.velocity[0]

    # During high -speed mode
    msg.buttons = [0] * _BUTTON_NUM
    for idx in params_fast[0]:
        msg.buttons[idx] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_vel = joy_ctrl.joint_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert len(joint_vel.name) == 1
    assert len(joint_vel.velocity) == 1
    assert joint_vel.name == [joint_name]
    assert pytest.approx(params_fast[2] * msg.axes[params_fast[1]]) == joint_vel.velocity[0]

    # Do not issue when the analog stick is input 0
    msg.axes[params[1]] = 0.0
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_vel = joy_ctrl.joint_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert not is_sub


def sleep(node: Node, time: float):
    rate = node.create_rate(1 / time, node.get_clock())
    rate.sleep()


# Multiple axis operating test
def test_multi_joint_control(setup):
    test_node = setup
    joy_ctrl = JoystickControlTest(test_node)
    assert joy_ctrl.joint_velocity_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

    # Get parameters
    joint_name1 = "head_pan_joint"
    joint_name2 = "head_tilt_joint"
    params = get_parameters(
        test_node,
        ["controls.head_control.button",
         "controls.head_control.dead_zone",
         f"controls.head_control.{joint_name1}.axis",
         f"controls.head_control.{joint_name2}.axis",
         f"controls.head_control.{joint_name1}.velocity",
         f"controls.head_control.{joint_name2}.velocity"])
    params_fast = get_parameters(
        test_node,
        ["controls.head_control_fast.button",
         "controls.head_control_fast.dead_zone",
         f"controls.head_control_fast.{joint_name1}.axis",
         f"controls.head_control_fast.{joint_name2}.axis",
         f"controls.head_control_fast.{joint_name1}.velocity",
         f"controls.head_control_fast.{joint_name2}.velocity"])

    # Manual creation of joystick MSG
    # If only one axis is input, only one axis is issued.
    msg = Joy()
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    msg.axes = [0.0] * _AXES_NUM
    msg.axes[params[3]] = 0.8
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_vel = joy_ctrl.joint_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert len(joint_vel.name) == 1
    assert len(joint_vel.velocity) == 1
    assert joint_vel.name == [joint_name2]
    assert pytest.approx(params[5] * msg.axes[params[3]]) == joint_vel.velocity[0]

    # If there is an input of both axis, the issuance is also issued.
    msg.axes[params[2]] = 0.6
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_vel = joy_ctrl.joint_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert len(joint_vel.name) == 2
    assert len(joint_vel.velocity) == 2
    assert joint_vel.name == [joint_name1, joint_name2]
    assert pytest.approx(params[4] * msg.axes[params[2]]) == joint_vel.velocity[0]
    assert pytest.approx(params[5] * msg.axes[params[3]]) == joint_vel.velocity[1]

    # During high -speed mode
    msg.buttons = [0] * _BUTTON_NUM
    for idx in params_fast[0]:
        msg.buttons[idx] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_vel = joy_ctrl.joint_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert len(joint_vel.name) == 2
    assert len(joint_vel.velocity) == 2
    assert joint_vel.name == [joint_name1, joint_name2]
    assert pytest.approx(params_fast[4] * msg.axes[params_fast[2]]) == joint_vel.velocity[0]
    assert pytest.approx(params_fast[5] * msg.axes[params_fast[3]]) == joint_vel.velocity[1]

    # If it does not exceed the unusual zone, it will not be issued (in high -speed mode)
    # Issued at the speed of the normal mode if it exceeds the non -sense of the normal mode
    dead_zone = params_fast[1]
    msg.axes = [0.0] * _AXES_NUM
    for idx in params_fast[2:4]:
        msg.axes[idx] = dead_zone - 0.01
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_vel = joy_ctrl.joint_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert len(joint_vel.name) == 2
    assert len(joint_vel.velocity) == 2
    assert joint_vel.name == [joint_name1, joint_name2]
    assert pytest.approx(params[4] * msg.axes[params[2]]) == joint_vel.velocity[0]
    assert pytest.approx(params[5] * msg.axes[params[3]]) == joint_vel.velocity[1]

    # If it does not exceed the unusual zone, it will not be issued (normal)
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    dead_zone = params[1]
    msg.axes = [0.0] * _AXES_NUM
    for idx in params[2:4]:
        msg.axes[idx] = dead_zone - 0.01
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_vel = joy_ctrl.joint_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert not is_sub


# Hand opening and closing operation test
def test_hand_control(setup):
    test_node = setup
    joy_ctrl = JoystickControlTest(test_node)
    assert joy_ctrl.gripper_control_grasp_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

    # Get parameters
    params = get_parameters(
        test_node,
        ["controls.hand_control.button",
         "controls.hand_control.force_enable_button",
         "controls.hand_control.open_angle_deg",
         "controls.hand_control.half_open_angle_deg"])

    # Hand close
    msg = Joy()
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, _ = joy_ctrl.gripper_control_grasp_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    sleep(test_node, _PUB_WAIT)

    # Open a hand
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_traj = joy_ctrl.gripper_control_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert pytest.approx(joint_traj.points[0].positions[0]) == math.radians(params[2])
    assert joint_traj.joint_names == ["hand_motor_joint"]
    sleep(test_node, _PUB_WAIT)

    # Close by grip control
    msg.buttons[params[1]] = 1
    # Measures in which Gripper_control moves first
    msg.axes = [0.0] * _AXES_NUM
    joy_ctrl.joy_pub.publish(msg)

    is_sub, _ = joy_ctrl.gripper_control_apply_force_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    sleep(test_node, _PUB_WAIT)

    # Hand open half of the hand
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_traj = joy_ctrl.gripper_control_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert pytest.approx(joint_traj.points[0].positions[0]) == math.radians(params[3])
    assert joint_traj.joint_names == ["hand_motor_joint"]


# Servo ON/OFF control operation test
def test_drive_power_control(setup):
    test_node = setup
    joy_ctrl = JoystickControlTest(test_node)
    assert joy_ctrl.command_drive_power_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

    # Get parameters
    params = get_parameters(
        test_node,
        ["controls.drive_power_control.button",
         "controls.drive_power_control.enable_button"])

    # Servo OFF (initial value is ON)
    msg = Joy()
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    msg.buttons[params[1]] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, servo_status = joy_ctrl.command_drive_power_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert not servo_status.data
    sleep(test_node, _PUB_WAIT)

    # Servo on
    joy_ctrl.joy_pub.publish(msg)

    is_sub, servo_status = joy_ctrl.command_drive_power_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert servo_status.data
    sleep(test_node, _PUB_WAIT)

    # Do not issue if enable_button is not pressed
    msg.buttons[params[1]] = 0
    joy_ctrl.joy_pub.publish(msg)

    is_sub, servo_status = joy_ctrl.command_drive_power_sub.wait_for_message(_SUB_TIMEOUT)
    assert not is_sub


# FIXME
# Suction operation test
# def test_suction_control(setup):
#     test_node = setup
#     joy_ctrl = JoystickControlTest(test_node)
#     assert joy_ctrl.suction_control_goal_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

#     # Get parameters
#     params = get_parameters(
#         test_node,
#         ["controls.suction_control.button"])

#     # Attract on
#     msg = Joy()
#     msg.buttons = [0] * _BUTTON_NUM
#     msg.buttons[params[0]] = 1
#     joy_ctrl.joy_pub.publish(msg)

#     is_sub, goal = joy_ctrl.suction_control_goal_sub.wait_for_message(_SUB_TIMEOUT)
#     assert is_sub
#     assert goal.suction_on.data
#     sleep(test_node, _PUB_WAIT)

#     # Attract off
#     joy_ctrl.joy_pub.publish(msg)

#     is_sub, goal = joy_ctrl.suction_control_goal_sub.wait_for_message(_SUB_TIMEOUT)
#     assert is_sub
#     assert not goal.suction_on.data


# FIXME
# Automatic charging docking operation test
# def test_auto_charge_dock_control(setup):
#     test_node = setup
#     joy_ctrl = JoystickControlTest(test_node)
#     assert joy_ctrl.autocharge_dock_control_goal_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

#     # Get parameters
#     params = get_parameters(
#         test_node,
#         ["controls.suction_control.button",
#          "controls.suction_control.enable_button"])

#     msg = Joy()
#     msg.buttons = [0] * _BUTTON_NUM
#     msg.buttons[params[0]] = 1
#     msg.buttons[params[1]] = 1
#     joy_ctrl.joy_pub.publish(msg)

#     is_sub, _ = joy_ctrl.autocharge_dock_control_goal_sub.wait_for_message(_SUB_TIMEOUT)
#     assert is_sub

#     # Do not issue if enable_button is not pressed
#     msg.buttons[params[1]] = 0
#     joy_ctrl.joy_pub.publish(msg)

#     is_sub, _ = joy_ctrl.autocharge_dock_control_goal_sub.wait_for_message(_SUB_TIMEOUT)
#     assert not is_sub


def _test_pose_control(joint_ref, names, positions):
    assert len(joint_ref.name) == len(names)
    assert len(joint_ref.position) == len(positions)
    assert joint_ref.name == names
    for i in range(len(positions)):
        assert pytest.approx(joint_ref.position[i]) == positions[i]


# Initial posture relocation movement test
def test_init_pose_control(setup):
    test_node = setup
    joy_ctrl = JoystickControlTest(test_node)
    assert joy_ctrl.joint_reference_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

    # Get parameters
    params = get_parameters(
        test_node,
        ["controls.init_pose.button",
         "controls.init_pose.joint_names",
         "controls.init_pose.joint_positions",
         "controls.pickup1_pose.button"])

    # Posture reset
    msg = Joy()
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_ref = joy_ctrl.joint_reference_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    _test_pose_control(joint_ref, params[1], params[2])

    # Posture reset (ignored if other buttons are pressed)
    msg.buttons[params[3]] = 1
    # If Controls.pickup1_pose.button is pressed, it will be haed_control, so add Axes
    msg.axes = [0.0] * _AXES_NUM
    joy_ctrl.joy_pub.publish(msg)

    is_sub, _ = joy_ctrl.joint_reference_sub.wait_for_message(_SUB_TIMEOUT)
    assert not is_sub


# Default posture relocation movement test
def test_default_pose_control(setup):
    test_node = setup
    joy_ctrl = JoystickControlTest(test_node)
    assert joy_ctrl.joint_reference_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

    # Get parameters
    params = get_parameters(
        test_node,
        ["controls.pickup1_pose.button",
         "controls.pickup1_pose.joint_names",
         "controls.pickup1_pose.joint_positions",
         "controls.pickup1_pose.enable_button"])

    # Migration of the established posture
    msg = Joy()
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    msg.buttons[params[3]] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_ref = joy_ctrl.joint_reference_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    _test_pose_control(joint_ref, params[1], params[2])

    # Do not issue if enable_button is not pressed
    msg.buttons[params[3]] = 0
    # If only Controls.pickup1_pose.button is pressed, it will be haed_control, so add Axes.
    msg.axes = [0.0] * _AXES_NUM
    joy_ctrl.joy_pub.publish(msg)

    is_sub, _ = joy_ctrl.command_drive_power_sub.wait_for_message(_SUB_TIMEOUT)
    assert not is_sub


# Dynamic preservation posture relocation movement test
def test_dynamic_storage_pose_control(setup):
    test_node = setup
    joy_ctrl = JoystickControlTest(test_node)
    assert joy_ctrl.joint_reference_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

    # Get parameters
    params = get_parameters(
        test_node,
        ["controls.dynamic_storage_pose.button",
         "controls.dynamic_storage_pose.joint_names",
         "controls.dynamic_storage_pose.joint_positions",
         "controls.dynamic_storage_pose.enable_button",
         "controls.dynamic_storage_pose.pose_save_button"])

    # Dynamic preservation posture transition (transition to the position set with parameters before saving)
    msg = Joy()
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    msg.buttons[params[3]] = 1
    msg.axes = [0.0] * _AXES_NUM
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_ref = joy_ctrl.joint_reference_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    _test_pose_control(joint_ref, params[1], params[2])

    # Change your posture and issue it
    joint_ref_save = [i * 0.1 for i in range(len(params[2]))]
    joint_ref.position = joint_ref_save
    joy_ctrl.joint_states_pub.publish(joint_ref)
    sleep(test_node, _PUB_WAIT)

    # Save the changed posture
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[3]] = 1
    msg.buttons[params[4]] = 1
    joy_ctrl.joy_pub.publish(msg)
    sleep(test_node, _PUB_WAIT)

    # Transition to dynamic preserved posture
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    msg.buttons[params[3]] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, joint_ref = joy_ctrl.joint_reference_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    _test_pose_control(joint_ref, params[1], joint_ref_save)

    # Do not issue if enable_button is not pressed
    msg.buttons[params[3]] = 0
    joy_ctrl.joy_pub.publish(msg)

    is_sub, _ = joy_ctrl.command_drive_power_sub.wait_for_message(_SUB_TIMEOUT)
    assert not is_sub


# Hand operation (including trolleys movement) Test
def test_endeffector_control(setup):
    test_node = setup
    joy_ctrl = JoystickControlTest(test_node)
    assert joy_ctrl.endeffector_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

    # Get parameters
    params = get_parameters(
        test_node,
        ["controls.endeffector_control.button",
         "controls.endeffector_control.dead_zone",
         "controls.endeffector_control.axis.x",
         "controls.endeffector_control.axis.y",
         "controls.endeffector_control.axis.z",
         "controls.endeffector_control.scale.x",
         "controls.endeffector_control.scale.y",
         "controls.endeffector_control.scale.z"])
    params_fast = get_parameters(
        test_node,
        ["controls.endeffector_control_fast.button",
         "controls.endeffector_control_fast.dead_zone",
         "controls.endeffector_control_fast.axis.x",
         "controls.endeffector_control_fast.axis.y",
         "controls.endeffector_control_fast.axis.z",
         "controls.endeffector_control_fast.scale.x",
         "controls.endeffector_control_fast.scale.y",
         "controls.endeffector_control_fast.scale.z"])

    # Manual creation of joystick MSG
    msg = Joy()
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    msg.axes = [0.0] * _AXES_NUM
    axes = [0.5, 0.8, -1.0]
    for i, idx in enumerate(params[2:5]):
        msg.axes[idx] = axes[i]
    joy_ctrl.joy_pub.publish(msg)

    is_sub, twist_stamped = joy_ctrl.endeffector_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert pytest.approx(params[5] * msg.axes[params[2]]) == twist_stamped.twist.linear.x
    assert pytest.approx(params[6] * msg.axes[params[3]]) == twist_stamped.twist.linear.y
    assert pytest.approx(params[7] * msg.axes[params[4]]) == twist_stamped.twist.linear.z

    # During high -speed mode
    msg.buttons = [0] * _BUTTON_NUM
    for idx in params_fast[0]:
        msg.buttons[idx] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, twist_stamped = joy_ctrl.endeffector_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert pytest.approx(params_fast[5] * msg.axes[params_fast[2]]) == twist_stamped.twist.linear.x
    assert pytest.approx(params_fast[6] * msg.axes[params_fast[3]]) == twist_stamped.twist.linear.y
    assert pytest.approx(params_fast[7] * msg.axes[params_fast[4]]) == twist_stamped.twist.linear.z

    # Speed ​​0 (in high -speed mode)
    # Issued at the speed of the normal mode if it exceeds the non -sense of the normal mode
    dead_zone = params_fast[1]
    msg.axes = [0.0] * _AXES_NUM
    for idx in params_fast[2:5]:
        msg.axes[idx] = dead_zone - 0.01
    joy_ctrl.joy_pub.publish(msg)

    # Issued in normal mode
    is_sub, twist_stamped = joy_ctrl.endeffector_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert pytest.approx(params[5] * msg.axes[params_fast[2]]) == twist_stamped.twist.linear.x
    assert pytest.approx(params[6] * msg.axes[params_fast[3]]) == twist_stamped.twist.linear.y
    assert pytest.approx(params[7] * msg.axes[params_fast[4]]) == twist_stamped.twist.linear.z

    # If it does not exceed the unusual zone, it will not be issued (normal)
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    dead_zone = params[1]
    msg.axes = [0.0] * _AXES_NUM
    for idx in params[2:5]:
        msg.axes[idx] = dead_zone - 0.01
    joy_ctrl.joy_pub.publish(msg)

    is_sub, twist_stamped = joy_ctrl.endeffector_sub.wait_for_message(_SUB_TIMEOUT)
    assert not is_sub


# Testing (including bogie movement) test
def test_endeffector_with_base_control(setup):
    test_node = setup
    joy_ctrl = JoystickControlTest(test_node)
    assert joy_ctrl.endeffector_with_base_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

    # Get parameters
    params = get_parameters(
        test_node,
        ["controls.endeffector_control.button",
         "controls.endeffector_control.dead_zone",
         "controls.endeffector_control.axis.x",
         "controls.endeffector_control.axis.y",
         "controls.endeffector_control.axis.z",
         "controls.endeffector_control.scale.x",
         "controls.endeffector_control.scale.y",
         "controls.endeffector_control.scale.z",
         "controls.endeffector_control.hand_with_base_button"])
    params_fast = get_parameters(
        test_node,
        ["controls.endeffector_control_fast.button",
         "controls.endeffector_control_fast.dead_zone",
         "controls.endeffector_control_fast.axis.x",
         "controls.endeffector_control_fast.axis.y",
         "controls.endeffector_control_fast.axis.z",
         "controls.endeffector_control_fast.scale.x",
         "controls.endeffector_control_fast.scale.y",
         "controls.endeffector_control_fast.scale.z",
         "controls.endeffector_control_fast.hand_with_base_button"])

    # Manual creation of joystick MSG
    msg = Joy()
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    msg.buttons[params[8]] = 1
    msg.axes = [0.0] * _AXES_NUM
    axes = [0.5, 0.8, -1.0]
    for i, idx in enumerate(params[2:5]):
        msg.axes[idx] = axes[i]
    joy_ctrl.joy_pub.publish(msg)

    is_sub, twist_stamped = joy_ctrl.endeffector_with_base_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert pytest.approx(params[5] * msg.axes[params[2]]) == twist_stamped.twist.linear.x
    assert pytest.approx(params[6] * msg.axes[params[3]]) == twist_stamped.twist.linear.y
    assert pytest.approx(params[7] * msg.axes[params[4]]) == twist_stamped.twist.linear.z

    # During high -speed mode
    msg.buttons = [0] * _BUTTON_NUM
    for idx in params_fast[0]:
        msg.buttons[idx] = 1
    msg.buttons[params[8]] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, twist_stamped = joy_ctrl.endeffector_with_base_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert pytest.approx(params_fast[5] * msg.axes[params_fast[2]]) == twist_stamped.twist.linear.x
    assert pytest.approx(params_fast[6] * msg.axes[params_fast[3]]) == twist_stamped.twist.linear.y
    assert pytest.approx(params_fast[7] * msg.axes[params_fast[4]]) == twist_stamped.twist.linear.z

    # Speed ​​0 (in high -speed mode)
    # Issued at the speed of the normal mode if it exceeds the non -sense of the normal mode
    dead_zone = params_fast[1]
    msg.axes = [0.0] * _AXES_NUM
    for idx in params_fast[2:5]:
        msg.axes[idx] = dead_zone - 0.01
    joy_ctrl.joy_pub.publish(msg)

    # Issued in normal mode
    is_sub, twist_stamped = joy_ctrl.endeffector_with_base_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert pytest.approx(params[5] * msg.axes[params_fast[2]]) == twist_stamped.twist.linear.x
    assert pytest.approx(params[6] * msg.axes[params_fast[3]]) == twist_stamped.twist.linear.y
    assert pytest.approx(params[7] * msg.axes[params_fast[4]]) == twist_stamped.twist.linear.z

    # If it does not exceed the unusual zone, it will not be issued (normal)
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    msg.buttons[params[8]] = 1
    dead_zone = params[1]
    msg.axes = [0.0] * _AXES_NUM
    for idx in params[2:5]:
        msg.axes[idx] = dead_zone - 0.01
    joy_ctrl.joy_pub.publish(msg)

    is_sub, twist_stamped = joy_ctrl.endeffector_with_base_sub.wait_for_message(_SUB_TIMEOUT)
    assert not is_sub


# Whether it is determined by sorting in order of the number of elements of button
def test_sort(setup):
    test_node = setup
    joy_ctrl = JoystickControlTest(test_node)
    assert joy_ctrl.joint_reference_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)
    assert joy_ctrl.command_velocity_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)
    assert joy_ctrl.joint_velocity_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

    # Get parameters
    params = get_parameters(
        test_node,
        ["controls.init_pose.button",
         "controls.button_length2.button",
         "controls.button_length2.axis.x",
         "controls.button_length3.button"])

    # Manual creation of joystick MSG
    msg = Joy()
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[0]] = 1
    msg.axes = [0.0] * _AXES_NUM
    sleep(test_node, _PUB_WAIT)
    joy_ctrl.joy_pub.publish(msg)

    # When one button is pressed
    is_sub, _ = joy_ctrl.joint_reference_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub

    # When two buttons are pressed at the same time
    msg.buttons = [0] * _BUTTON_NUM
    for idx in params[1]:
        msg.buttons[idx] = 1
    msg.axes[params[2]] = 1.0
    joy_ctrl.joy_pub.publish(msg)

    # Two elements of button are given priority
    is_sub, _ = joy_ctrl.command_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub

    # When 3 buttons are pressed at the same time
    msg.buttons = [0] * _BUTTON_NUM
    for idx in params[3]:
        msg.buttons[idx] = 1
    joy_ctrl.joy_pub.publish(msg)

    # Two elements of button are given priority
    is_sub, _ = joy_ctrl.joint_velocity_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub


# Voice notification test
def test_voice_notification(setup):
    test_node = setup
    joy_ctrl = JoystickControlTest(test_node)
    assert joy_ctrl.talk_request_sub.wait_until_connection_established(_CONNECTION_TIMEOUT)

    # Get parameters
    control1 = "endeffector_control"
    control2 = "head_control"
    params = get_parameters(
        test_node,
        [f"controls.{control1}.button",
         f"controls.{control1}.axis.x",
         f"controls.{control1}.axis.y",
         f"controls.{control1}.axis.z",
         f"controls.{control2}.button",
         "controls.hand_control.button",
         "controls.init_pose.button"])

    # Reset the state
    msg = Joy()
    msg.buttons = [0] * _BUTTON_NUM
    joy_ctrl.joy_pub.publish(msg)

    # Voice notification if notification is valid
    # endeffector_control
    msg.buttons[params[0]] = 1
    msg.axes = [0.0] * _AXES_NUM
    for idx in params[1:3]:
        msg.axes[idx] = 1.0
    joy_ctrl.joy_pub.publish(msg)

    is_sub, voice = joy_ctrl.talk_request_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert voice.sentence == control1

    # head_control
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[4]] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, voice = joy_ctrl.talk_request_sub.wait_for_message(_SUB_TIMEOUT)
    assert is_sub
    assert voice.sentence == control2

    # If the notification is invalid
    # hand_control
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[5]] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, voice = joy_ctrl.talk_request_sub.wait_for_message(_SUB_TIMEOUT)
    assert not is_sub

    # init_pose
    msg.buttons = [0] * _BUTTON_NUM
    msg.buttons[params[6]] = 1
    joy_ctrl.joy_pub.publish(msg)

    is_sub, voice = joy_ctrl.talk_request_sub.wait_for_message(_SUB_TIMEOUT)
    assert not is_sub


# If there is no necessary parameters
def test_no_type():
    rclpy.init()
    joy_config_path = os.path.join(
        get_package_share_directory("hsrb_joystick_teleop"),
        "config",
        "test_joystick_control_config.yaml")
    node = rclpy.create_node(
        "node",
        cli_args=["--ros-args", "--params-file", joy_config_path],
        allow_undeclared_parameters=True,
        automatically_declare_parameters_from_overrides=True)
    DrivePowerControl("no_type", node)
    assert not rclpy.ok()


# If there is no Controlls
def test_no_control():
    rclpy.init()
    joy_config_path = os.path.join(
        get_package_share_directory("hsrb_joystick_teleop"),
        "config",
        "test_no_control_config.yaml")
    JoystickControlManager(joy_config_path)
    assert not rclpy.ok()
