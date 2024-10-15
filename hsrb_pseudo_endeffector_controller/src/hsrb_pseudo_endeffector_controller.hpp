/*
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
*/
#ifndef HSRB_PSEUDO_ENDEFFECTOR_CONTROLLER_HSRB_PSEUDO_ENDEFFECTOR_CONTROLLER_HPP
#define HSRB_PSEUDO_ENDEFFECTOR_CONTROLLER_HSRB_PSEUDO_ENDEFFECTOR_CONTROLLER_HPP

#include <optional>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <tmc_robot_kinematics_model/pinocchio_wrapper.hpp>

#include <tmc_manipulation_types/manipulation_types.hpp>
#include <tmc_manipulation_util/joint_trajectory_publisher.hpp>
#include <tmc_robot_kinematics_model/numeric_ik_solver.hpp>

namespace hsrb_pseudo_endeffector_controller {

// Pseudo -hand control controller class
class PseudoEndeffectorController : public rclcpp::Node {
 public:
  PseudoEndeffectorController();
  explicit PseudoEndeffectorController(const rclcpp::NodeOptions& options);

 private:
  // Update the state of the robot from the latest Joint_states, ODOM
  bool UpdateJointState();
  // Publish Command from the current robot status
  void PublishJointCommand(const rclcpp::Time& stamp, double duration) const;
  // Calculate the state of the robot after moving
  bool CalcNextState(const tmc_manipulation_types::BaseMovementType& base_type,
                     const Eigen::Affine3d& origin_to_next_end);

  // Command_velocity -based callback common processing
  void VelocityCallback(const geometry_msgs::msg::TwistStamped::SharedPtr& command,
                        const tmc_manipulation_types::BaseMovementType& base_type);

  void CommandVelocityCallback(const geometry_msgs::msg::TwistStamped::SharedPtr command);
  void CommandVelocityWithBaseCallback(const geometry_msgs::msg::TwistStamped::SharedPtr command);

  void JointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr state) { latest_joint_states_ = state; }
  void OdomCallback(const nav_msgs::msg::Odometry::SharedPtr odom) { latest_odom_ = odom; }

  // Publish a success or fail
  void PublishIsSuccess(bool result);

  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr command_velocity_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr command_velocity_with_base_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  sensor_msgs::msg::JointState::SharedPtr latest_joint_states_;
  nav_msgs::msg::Odometry::SharedPtr latest_odom_;

  std::vector<tmc_manipulation_util::JointTrajectoryPublisher::Ptr> arm_command_pubs_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr base_command_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr success_pub_;

  // Robot model
  tmc_robot_kinematics_model::IRobotKinematicsModel::Ptr robot_;
  // IK Solva
  tmc_robot_kinematics_model::IKSolver::Ptr ik_solver_;

  // EndefFFECTOR frame name
  std::string endeffector_frame_name_;
  // Operating joints
  std::vector<std::string> use_joints_;
  // IK joint weight
  Eigen::VectorXd ik_arm_weights_;
  Eigen::VectorXd ik_base_weights_;
  // Interpolation time during speed control
  double velocity_duration_;

  // EndefFECTOR posture
  Eigen::Affine3d origin_to_end_;
  // Attachment of the command value criterion frame
  Eigen::Affine3d origin_to_base_;
  // The previous command value
  geometry_msgs::msg::Twist last_command_value_;
  // Time when the previous command flew
  rclcpp::Time last_command_stamp_;
  // Standard frame in the previous command
  std::string last_command_frame_;
  // Skilled values ​​that determine that a continuous command has been interrupted [SEC]
  double discontinuous_period_;
  // With true, if the command is not interrupted, the current joint status does not feedback.
  bool open_loop_control_;
  // Publishing period[s]
  std::optional<double> publish_period_;

  Eigen::VectorXd GetWeightParameter(const std::string& parameter_name, uint32_t dof);
};

}  // namespace hsrb_pseudo_endeffector_controller

#endif  // HSRB_PSEUDO_ENDEFFECTOR_CONTROLLER_HSRB_PSEUDO_ENDEFFECTOR_CONTROLLER_HPP
