#!/usr/bin/env python3
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
# -*- coding: utf-8 -*-
from abc import (
    ABC,
    abstractmethod,
)
import math
from operator import itemgetter
import os
from typing import (
    Any,
    Optional,
    Tuple,
)

from geometry_msgs.msg import (
    Twist,
    TwistStamped,
)
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import (
    JointState,
    Joy,
)
from std_msgs.msg import Bool
from tmc_control_msgs.action import GripperApplyEffort
from tmc_msgs.msg import JointVelocity
from tmc_voice_msgs.msg import Voice
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)

# FIXME
# from tmc_suction.action import SuctionControl as SuctionControlAction
# from hsrb_autocharge.action import DockChargeStation

# JOYSTICK-> Coordinate name table for twist conversion
_AXIS_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]
# Whether there is a voice notification
_DEFAULT_VOICE_NOTIFICATION = False
# Time until the next command is acceptable [S]
_COMMAND_IGNORE_TIME = 0.5
# Torque to close the hand [NM]
_HAND_CLOSE_TORQUE = -0.018
# Power to close the hand (force control) [N]
_HAND_CLOSE_FORCE = 0.8
# Time required to open a hand
_OPEN_TIME_FROM_START = 1.0
# Hand joint name
_HAND_JOINT_NAME = "hand_motor_joint"
# Timeout time of suction action [S]
_SUCTION_TIMEOUT = 5.0


def clamp(value: float, smallest: float, largest: float):
    """Make the value saturated to the upper and lower limit.

    Args:a
        value (float): 値
        Smallest (Float): Small value
        Largest (Float): Maximum value

    Returns:
        Float: saturated value
    """
    return min(max(value, smallest), largest)


class JoystickControl(ABC):
    """Joystic control base class."""

    def __init__(self, name: str, node: Node):
        self._name = name
        self._node = node
        self._voice_notification = self.get_param("voice_notification")

        button = self.get_param("button")
        self._buttons = []
        if isinstance(button, int):
            self._buttons = [button]
        elif isinstance(button, (tuple, list)):
            self._buttons = list(button)
        else:
            raise TypeError(
                f"Parameter 'button' does not support {type(button)}. "
                + "It should be specified as 'int', 'list', or 'tuple'.")

    def __len__(self):
        return len(self._buttons)

    @abstractmethod
    def update(self) -> bool:
        """Update to operation."""
        ...

    def get_param(self, key: str) -> Any:
        """Acquisition of parameters."""
        value = self._node.get_parameter(f"{self._name}.{key}").value
        if value is None:
            self._node.get_logger().fatal(f"Could not get parameter: {self._name}.{key}.")
            self._node.destroy_node()
            rclpy.shutdown()
        return value

    def get_params_by_prefix(self, prefix: str) -> Any:
        """PREFIX below obtained parameter names."""
        return self._node.get_parameters_by_prefix(f"{self._name}.{prefix}")

    def get_name(self) -> str:
        """Acquire the control name."""
        return self._name

    def get_notification(self) -> bool:
        """Acquisition of audio notification."""
        return self._voice_notification

    def check_all_buttons_pressed(self, msg: Joy) -> bool:
        """Judgment is whether all the specified buttons are pressed."""
        buttons = itemgetter(*self._buttons)(msg.buttons)
        if isinstance(buttons, int):
            return buttons == 1
        else:
            return all(buttons)


class JointControl(JoystickControl):
    """Class that controls joints to 2 axes."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        self._dead_zone = 0.0
        self._joint_velocity_pub = node.create_publisher(
            JointVelocity,
            "pseudo_velocity_controller/ref_joint_velocity",
            1)
        # Acquisition of parameters
        self._joint_settings = self._node.get_parameters_by_prefix(self._name)
        # Obtain the joint name to be operated
        self._target_joints = []
        for name in self._joint_settings.keys():
            if "joint" in name:
                joint_name = name.split(".")[0]
                if joint_name not in self._target_joints:
                    self._target_joints.append(joint_name)

    def update(self, msg: Joy) -> bool:
        if self.check_all_buttons_pressed(msg):
            joint_velocity = JointVelocity()
            joint_velocity.header.stamp = self._node.get_clock().now().to_msg()
            for name in self._target_joints:
                axis = self._joint_settings[f"{name}.axis"].value
                velocity = self._joint_settings[f"{name}.velocity"].value
                scale_factor = 0.0
                if axis >= 0 and len(msg.axes) > axis:
                    scale_factor = clamp(msg.axes[axis], -1.0, 1.0)
                # Ignore if it is not exceeding the unusual zone
                if math.fabs(scale_factor) > self._dead_zone:
                    joint_velocity.name.append(name)
                    joint_velocity.velocity.append(velocity * scale_factor)
            # publish
            if joint_velocity.name != []:
                self._joint_velocity_pub.publish(joint_velocity)
                return True
        return False


class SingleJointControl(JointControl):
    """Class to control one axis."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)


class MultiJointControl(JointControl):
    """2 -axis joint control class."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        self._dead_zone = self.get_param("dead_zone")


class TwistControl(JoystickControl):
    """Twist operation control class."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        self._dead_zone = self.get_param("dead_zone")

        # Reading AXIS parameters
        self._axis_map = {}
        for name, param in self.get_params_by_prefix("axis").items():
            if name not in _AXIS_NAMES:
                continue
            self._axis_map[name] = param.value

        # Reading Scale parameters
        self._scale_map = {}
        for name, param in self.get_params_by_prefix("scale").items():
            if name not in _AXIS_NAMES:
                continue
            self._scale_map[name] = param.value

    def calc_twist(self, msg: Joy) -> Tuple[bool, Twist]:
        """Twist calculation ..."""
        # Output if you are all in the unusual zone or at least one.
        twist_out = Twist()
        is_twist_output_valid = False
        for name, axis in self._axis_map.items():
            scale_factor = clamp(msg.axes[axis], -1.0, 1.0)
            if math.fabs(scale_factor) < self._dead_zone:
                scale_factor = 0.0
            else:
                scale = self._scale_map.get(name)
                if scale is None:
                    raise KeyError(f"'{name}' does not exist in the '{self._name}.scale'.")
                scale_factor *= scale
                is_twist_output_valid = True

            twist_list = {"x": "twist_out.linear.x",
                          "y": "twist_out.linear.y",
                          "z": "twist_out.linear.z",
                          "roll": "twist_out.angular.x",
                          "pitch": "twist_out.angular.y",
                          "yaw": "twist_out.angular.z"}
            exec(f"{twist_list[name]} = {scale_factor}")

        return is_twist_output_valid, twist_out


class BaseControl(TwistControl):
    """Class to control trolleys."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        self._command_velocity_pub = self._node.create_publisher(
            Twist, "command_velocity", 1)

        self._is_published = False

    @property
    def is_published(self):
        return self._is_published

    def update(self, msg: Joy) -> bool:
        if self.check_all_buttons_pressed(msg):
            # The return value is not determined because it is issued even when the speed is 0
            # Because I want to stop the trolley immediately when the input is gone
            _, twist = self.calc_twist(msg)
            self._command_velocity_pub.publish(twist)
            self._is_published = True
            return True
        else:
            # When the bogie movement valid button is separated, set the speed to 0
            if self._is_published:
                twist = Twist()
                self._command_velocity_pub.publish(twist)
                self._is_published = False
                return True
        return False


class OneTimeControl(JoystickControl):
    """Temporary execution class."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        self._last_control_time = None

    def update(self, msg: Joy) -> bool:
        now = self._node.get_clock().now()
        if self._last_control_time is not None:
            if (now - self._last_control_time) < Duration(seconds=_COMMAND_IGNORE_TIME):
                return False
        # Be sure to execute the first time
        updated = self.update_once(msg)
        if updated:
            self._last_control_time = now
        return updated

    @abstractmethod
    def update_once(self, msg: Joy) -> bool:
        ...


class HandControl(OneTimeControl):
    """Hand opening and closing execution class."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        self._hand_open = True
        self._apply_force_client = ActionClient(
            self._node, GripperApplyEffort, "gripper_controller/apply_force")
        self._grasp_client = ActionClient(
            self._node, GripperApplyEffort, "gripper_controller/grasp")
        self._gripper_trajectory_pub = self._node.create_publisher(
            JointTrajectory, "gripper_controller/joint_trajectory", 1)
        self._force_enable_button = self.get_param("force_enable_button")
        self._open_angle = math.radians(self.get_param("open_angle_deg"))
        self._half_open_angle = math.radians(self.get_param("half_open_angle_deg"))

    def _close(self, action_client: ActionClient, effort: float) -> None:
        hand_command = GripperApplyEffort.Goal()
        hand_command.effort = effort
        action_client.send_goal_async(hand_command)

    def _open(self, positions: float) -> None:
        point = JointTrajectoryPoint()
        point.positions.append(positions)
        point.time_from_start = Duration(seconds=_OPEN_TIME_FROM_START).to_msg()
        hand_trajectory = JointTrajectory()
        hand_trajectory.header.stamp = self._node.get_clock().now().to_msg()
        hand_trajectory.joint_names.append(_HAND_JOINT_NAME)
        hand_trajectory.points.append(point)
        self._gripper_trajectory_pub.publish(hand_trajectory)

    def update_once(self, msg: Joy) -> bool:
        if self.check_all_buttons_pressed(msg):
            if msg.buttons[self._force_enable_button] == 1:
                # Grip -in control
                if self._hand_open:
                    self._close(self._apply_force_client, _HAND_CLOSE_FORCE)
                else:
                    # Open only half
                    self._open(self._half_open_angle)
            else:
                # Hand opening and closing
                if self._hand_open:
                    self._close(self._grasp_client, _HAND_CLOSE_TORQUE)
                else:
                    self._open(self._open_angle)
            self._hand_open = not self._hand_open
            return True
        return False


class DrivePowerControl(OneTimeControl):
    """Servo ON/OFF control class."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        self._servo_on = True
        self._command_drive_power_pub = self._node.create_publisher(
            Bool, "command_drive_power", 1)
        self._servo_enable_button = self.get_param("enable_button")

    def update_once(self, msg: Joy) -> bool:
        if self.check_all_buttons_pressed(msg) and msg.buttons[self._servo_enable_button] == 1:
            # Servo command
            # Inverse the state
            self._servo_on = not self._servo_on
            servo_status = Bool()
            servo_status.data = self._servo_on
            self._command_drive_power_pub.publish(servo_status)
            return True
        return False


class SuctionControl(OneTimeControl):
    """Success execution class."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        self._suction_on = False
        # FIXME
        # self._suction_client = ActionClient(
        #     self._node, SuctionControlAction, "suction_control")
        # self._suction_client.wait_for_server()

    def update_once(self, msg: Joy) -> bool:
        if self.check_all_buttons_pressed(msg):
            # attract
            # Inverse the state
            self._suction_on = not self._suction_on
            # FIXME
            # suction_command = SuctionControlAction.Goal()
            # suction_command.timeout = Duration(seconds=_SUCTION_TIMEOUT).to_msg()
            # suction_command.suction_on.data = self._suction_on
            # self._suction_client.send_goal(suction_command)
            return True
        return False


class AutoChargeDockControl(OneTimeControl):
    """Automatic charging docking execution class."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        # self._dock_client = ActionClient( # FIXME
        #     self._node, DockChargeStation, "/hsrb/autocharge_node/dock")
        self._dock_enable_button = self.get_param("enable_button")

    def update_once(self, msg: Joy) -> bool:
        if self.check_all_buttons_pressed(msg) and msg.buttons[self._dock_enable_button] == 1:
            # dock_command = DockChargeStation.Goal() # FIXME
            # self._dock_client.send_goal(dock_command)
            return True
        return False


class EndeffectorControl(TwistControl):
    """Class that controls hand movement."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        self._hand_velocity_pub = self._node.create_publisher(
            TwistStamped, "pseudo_endeffector_controller/command_velocity", 1)
        self._hand_velocity_with_base_pub = self._node.create_publisher(
            TwistStamped, "pseudo_endeffector_controller/command_velocity_with_base", 1)

        self._frame = self.get_param("frame")
        self._hand_with_base_button = self.get_param("hand_with_base_button")

    def update(self, msg: Joy) -> bool:
        if self.check_all_buttons_pressed(msg):
            twist_stamped = TwistStamped()
            # Do not issue if there is no input or all within a non -sensitive zone
            is_valid, twist_stamped.twist = self.calc_twist(msg)
            if is_valid:
                twist_stamped.header.frame_id = self._frame
                # When including a trolley movement
                if msg.buttons[self._hand_with_base_button] == 1:
                    self._hand_velocity_with_base_pub.publish(twist_stamped)
                else:
                    self._hand_velocity_pub.publish(twist_stamped)
                return True
        return False


class PoseControl(OneTimeControl):
    """Stock transition class."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        self._safe_pose_changer_pub = self._node.create_publisher(
            JointState, "safe_pose_changer/joint_reference", 1)

        self._joint_names = self.get_param("joint_names")
        self._joint_potisions = self.get_param("joint_positions")

    def change_pose(self):
        joint_state = JointState()
        joint_state.name = self._joint_names
        joint_state.position = self._joint_potisions
        self._safe_pose_changer_pub.publish(joint_state)


class InitPoseControl(PoseControl):
    """Reset posture transition class."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

    def update_once(self, msg: Joy) -> bool:
        if self.check_all_buttons_pressed(msg):
            # Confirm that other buttons are not pushed at the same time
            # Because the dynamic preservation of the posture and the button are covered
            pushed_buttons = [pushed == 1 and i not in self._buttons
                              for i, pushed in enumerate(msg.buttons)]
            if any(pushed_buttons):
                return False
            self.change_pose()
            return True
        return False


class DefaultPoseControl(PoseControl):
    """Standard posture transition class."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        self._pose_enable_button = self.get_param("enable_button")

    def update_once(self, msg: Joy) -> bool:
        if self.check_all_buttons_pressed(msg) and msg.buttons[self._pose_enable_button] == 1:
            self.change_pose()
            return True
        return False


class DynamicStoragePoseControl(PoseControl):
    """Dynamic preservation posture transition class."""

    def __init__(self, name: str, node: Node):
        super().__init__(name, node)

        self._joint_states_sub = self._node.create_subscription(
            JointState,
            "joint_states",
            self.joint_state_callback,
            1)
        self._pose_enable_button = self.get_param("enable_button")
        self._pose_save_button = self.get_param("pose_save_button")
        self._joint_states = None
        self._save_pose = self._joint_potisions.copy()

    def joint_state_callback(self, msg: JointState):
        self._joint_states = msg

    def update_once(self, msg: Joy) -> bool:
        if msg.buttons[self._pose_enable_button] == 1 and msg.buttons[self._pose_save_button] == 1:
            # Dynamic preservation of default posture
            # Search in the order of joint name used in the prescribed posture, and acquire the position of the corresponding joint
            if self._joint_states is None:
                self._node.get_logger().info("joint_state has not been received yet.")
                return False
            self._save_pose = [
                self._joint_states.position[self._joint_states.name.index(joint_name)]
                for joint_name in self._joint_names]
            return True
        elif msg.buttons[self._pose_enable_button] == 1 and self.check_all_buttons_pressed(msg):
            joint_state = JointState()
            joint_state.name = self._joint_names
            joint_state.position = self._save_pose
            self._safe_pose_changer_pub.publish(joint_state)
            return True
        return False


class JoystickControlManager(Node):
    """Joystic control management class."""

    def __init__(self, joy_config_path: Optional[str] = None):
        if joy_config_path is None:
            super().__init__(
                "joystick_control_node",
                allow_undeclared_parameters=True,
                automatically_declare_parameters_from_overrides=True)
        else:
            if not os.path.isfile(joy_config_path):
                raise TypeError(f"File '{joy_config_path}' does not exist. ")
            super().__init__(
                "joystick_control_node",
                cli_args=["--ros-args", "--params-file", joy_config_path],
                allow_undeclared_parameters=True,
                automatically_declare_parameters_from_overrides=True)

        self._joy_sub = self.create_subscription(
            Joy, "joy", self.joy_callback, qos_profile_sensor_data)

        self._talk_pub = self.create_publisher(
            Voice, "talk_request", 1)

        self._voice_notification_on = _DEFAULT_VOICE_NOTIFICATION
        if self.has_parameter("assisted_voice_notification"):
            self._voice_notification_on = self.get_parameter("assisted_voice_notification")

        self._pre_mode = "None"

        joystick_config = self.get_parameters_by_prefix("controls")

        # Generate an instance according to Type when initializing
        controls = []
        for name, _type in joystick_config.items():
            split_name = name.split(".")
            if split_name[-1] == "type":
                ns = f"controls.{split_name[0]}"
                ctrl_type = _type.value
                if ctrl_type == "single_joint_control":
                    controls.append(SingleJointControl(ns, self))
                elif ctrl_type == "multi_joint_control":
                    controls.append(MultiJointControl(ns, self))
                elif ctrl_type == "base_control":
                    controls.append(BaseControl(ns, self))
                elif ctrl_type == "endeffector_control":
                    controls.append(EndeffectorControl(ns, self))
                elif ctrl_type == "init_pose_control":
                    controls.append(InitPoseControl(ns, self))
                elif ctrl_type == "dynamic_storage_pose_control":
                    controls.append(DynamicStoragePoseControl(ns, self))
                elif ctrl_type == "pose_control":
                    controls.append(DefaultPoseControl(ns, self))
                elif ctrl_type == "hand_control":
                    controls.append(HandControl(ns, self))
                elif ctrl_type == "autocharge_control":
                    controls.append(AutoChargeDockControl(ns, self))
                elif ctrl_type == "drive_power_control":
                    controls.append(DrivePowerControl(ns, self))
                elif ctrl_type == "suction_control":
                    controls.append(SuctionControl(ns, self))
                else:
                    self.get_logger().warn(f"{ctrl_type} is invalid control type. Ignored.")
                    continue
        if len(controls) == 0:
            self.get_logger().fatal("Set controller to joystick_control/controls at least 1.")
            self.destroy_node()
            rclpy.shutdown()
        # Sort in order of high elements of buttons_
        self._controls = sorted(controls, key=len, reverse=True)

    def joy_callback(self, msg: Joy) -> None:
        """Joystic input callback."""
        for c in self._controls:
            # Call the Update of each controller and do not carry out other operations if it is operated.
            if c.update(msg):
                # When the controller changes, notify with voice (when audio notification is valid)
                mode = c.get_name()
                if self._pre_mode != mode and c.get_notification():
                    self.voice_publish(mode)
                self._pre_mode = mode
                break

    def voice_publish(self, sentence: str) -> None:
        """Dprial."""
        if self._voice_notification_on:
            voice = Voice()
            voice.sentence = sentence.replace("controls.", "")
            if os.getenv("LANG") == "ja_JP.UTF-8":
                voice.language = Voice.JAPANESE
            else:
                voice.language = Voice.ENGLISH
            self._talk_pub.publish(voice)
