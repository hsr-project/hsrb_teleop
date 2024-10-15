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
#include "hsrb_pseudo_endeffector_controller.hpp"

#include <limits>
#include <memory>
#include <string>

#include <tmc_manipulation_types/utils.hpp>
#include <tmc_utils/parameters.hpp>
#include <tmc_utils/qos.hpp>

namespace {
// Numerical IK tolerance error
const double kDefaultIKDelta = 1.0e-3;
// Supplementary time during speed control
const double kDefaultVelocityDuration = 0.5;
// Numerical IK maximum number of repetitions
const int32_t kMaxItrIK = 1000;
// Numerical IK tolerance fluctuation
const double kIKConvergeThreshold = 1.0e-10;
// Skilled value that judges that the command has been interrupted [SEC]
const double kDefaultDiscontinuousPeriod = 0.5;
// Hand frame name default value
const char* const kDefaultEndEffectorFrame = "hand_palm_link";
// Default value of joint name used for IK
const std::vector<std::string> kDefaultUseJoints = {"arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
                                                    "wrist_flex_joint", "wrist_roll_joint"};
// The default value of the arm track controller
const std::vector<std::string> kTrajectoryControllers = {"arm_trajectory_controller"};

// Make twist to Transform (duration)
void TwistToTransform(const geometry_msgs::msg::Twist& twist,
                      double duration,
                      Eigen::Affine3d& transform_out) {
  Eigen::Quaterniond zero = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond angular =
      Eigen::AngleAxisd(twist.angular.x, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(twist.angular.y, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(twist.angular.z, Eigen::Vector3d::UnitZ());
  // Angular is an interpolation with SLERP
  transform_out = zero.slerp(duration, angular);
  // LINEAR just applies duration
  transform_out.translation() <<
      twist.linear.x * duration,
      twist.linear.y * duration,
      twist.linear.z * duration;
}

void CalcOriginToNextEnd(const Eigen::Affine3d& origin_to_end,
                         const Eigen::Affine3d& origin_to_frame,
                         const Eigen::Affine3d& transform,
                         Eigen::Affine3d& dst_origin_to_next_end) {
  // Endeffector-> Coordinates to reference frames
  Eigen::Affine3d end_to_frame = origin_to_end.inverse() * origin_to_frame;
  // Match the origin
  end_to_frame.translation() << 0.0, 0.0, 0.0;
  // Convert Transform, the origin of the reference frame, to the origin of Endeffector
  Eigen::Affine3d end_on_transform = end_to_frame * transform * end_to_frame.inverse();
  // Find the position of the following ENDEFFECTOR
  dst_origin_to_next_end = origin_to_end * end_on_transform;
}
}  // anonymous namespace

namespace hsrb_pseudo_endeffector_controller {

PseudoEndeffectorController::PseudoEndeffectorController() : PseudoEndeffectorController(rclcpp::NodeOptions()) {}

PseudoEndeffectorController::PseudoEndeffectorController(const rclcpp::NodeOptions& options)
    : Node("pseudo_endeffector_controller_node", options), last_command_stamp_(now()), last_command_frame_("") {
  velocity_duration_ = tmc_utils::GetParameter(this, "velocity_duration", kDefaultVelocityDuration);
  discontinuous_period_ = tmc_utils::GetParameter(this, "discontinuous_period", kDefaultDiscontinuousPeriod);
  endeffector_frame_name_ = tmc_utils::GetParameter(this, "end_effector_frame", kDefaultEndEffectorFrame);
  use_joints_ = tmc_utils::GetParameter(this, "use_joints", kDefaultUseJoints);
  ik_arm_weights_ = GetWeightParameter("ik_arm_weights", use_joints_.size());
  // The bogie is assumed to be 3 degree of freedom of ODOM_X/Y/T
  ik_base_weights_ = GetWeightParameter("ik_base_weights", 3);
  // Joint_trajectory_controller was named consciously
  open_loop_control_ = tmc_utils::GetParameter(this, "open_loop_control", false);

  double publish_rate = tmc_utils::GetParameter(this, "publish_rate", 30.0);
  if (publish_rate > 0.0) {
    publish_period_ = 1.0 / publish_rate;
  }

  const auto robot_description = tmc_utils::GetParameter<std::string>(this, "robot_description", "");
  robot_ = std::make_shared<tmc_robot_kinematics_model::PinocchioWrapper>(robot_description);

  const auto ik_delta = tmc_utils::GetParameter(this, "ik_delta", kDefaultIKDelta);
  ik_solver_ = std::make_shared<tmc_robot_kinematics_model::NumericIKSolver>(
      tmc_robot_kinematics_model::IKSolver::Ptr(), robot_, kMaxItrIK, ik_delta, kIKConvergeThreshold);

  // Publisher creation
  const auto controllers = tmc_utils::GetParameter(this, "arm_controllers", kTrajectoryControllers);
  for (const auto& controller_name : controllers) {
    arm_command_pubs_.push_back(
        std::make_shared<tmc_manipulation_util::JointTrajectoryPublisher>(this, controller_name));
  }
  base_command_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "omni_base_controller/joint_trajectory", tmc_utils::ReliableVolatileQoS());
  success_pub_ = create_publisher<std_msgs::msg::Bool>("~/success", tmc_utils::ReliableVolatileQoS());

  // Subscriper creation
  // The command value is a prerequisite that will fly continuously, so make it to Best_effort.
  command_velocity_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      "~/command_velocity", tmc_utils::BestEffortQoS(),
      std::bind(&PseudoEndeffectorController::CommandVelocityCallback, this, std::placeholders::_1));
  command_velocity_with_base_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      "~/command_velocity_with_base", tmc_utils::BestEffortQoS(),
      std::bind(&PseudoEndeffectorController::CommandVelocityWithBaseCallback, this, std::placeholders::_1));

  joint_states_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", tmc_utils::BestEffortQoS(),
      std::bind(&PseudoEndeffectorController::JointStatesCallback, this, std::placeholders::_1));
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "odom", tmc_utils::BestEffortQoS(),
      std::bind(&PseudoEndeffectorController::OdomCallback, this, std::placeholders::_1));
}

// Update the state of the robot from the latest Joint_states, ODOM
bool PseudoEndeffectorController::UpdateJointState() {
  if (!latest_joint_states_ || !latest_odom_) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5 * 1000, "joint_states or odom is not subscribed yet");
    return false;
  }

  // Convert the contents of JOINT_STATES to JointState type
  tmc_manipulation_types::JointState joint_state;
  // TODO(Takeshita) tmc_manipulation_types_bridgeのROS2化
  joint_state.name = latest_joint_states_->name;
  joint_state.position =
      Eigen::VectorXd::Map(latest_joint_states_->position.data(), latest_joint_states_->position.size());

  // Convert the content of ODOM to Eigen :: affinE3D type
  Eigen::Affine3d origin_to_base;
  origin_to_base = Eigen::Quaterniond(latest_odom_->pose.pose.orientation.w,
                                      latest_odom_->pose.pose.orientation.x,
                                      latest_odom_->pose.pose.orientation.y,
                                      latest_odom_->pose.pose.orientation.z);
  origin_to_base.translation() <<
      latest_odom_->pose.pose.position.x,
      latest_odom_->pose.pose.position.y,
      latest_odom_->pose.pose.position.z;

  // renew
  robot_->SetRobotTransform(origin_to_base);
  robot_->SetNamedAngle(tmc_manipulation_types::ExtractPartialJointState(joint_state, use_joints_));
  return true;
}

// Publish Command from the current robot status
void PseudoEndeffectorController::PublishJointCommand(const rclcpp::Time& stamp, double duration) const {
  // Robot_'s posture is updated
  const tmc_manipulation_types::JointState joint_state_out = robot_->GetNamedAngle(use_joints_);
  const int32_t num_joints = use_joints_.size();

  // Create JointTrajectory for ARM
  trajectory_msgs::msg::JointTrajectory arm_trajectory;
  arm_trajectory.joint_names = use_joints_;
  arm_trajectory.points.resize(1);
  arm_trajectory.points[0].positions.resize(num_joints);
  arm_trajectory.points[0].velocities.resize(num_joints);
  for (int i = 0; i < num_joints; ++i) {
    arm_trajectory.points[0].positions[i] = joint_state_out.position[i];
    arm_trajectory.points[0].velocities[i] = 0.0;
  }
  arm_trajectory.points[0].time_from_start = rclcpp::Duration::from_seconds(duration);

  // Create JointTrajectory for Base
  trajectory_msgs::msg::JointTrajectory base_trajectory;
  base_trajectory.joint_names.push_back("odom_x");
  base_trajectory.joint_names.push_back("odom_y");
  base_trajectory.joint_names.push_back("odom_t");
  base_trajectory.points.resize(1);
  base_trajectory.points[0].positions.resize(3);
  base_trajectory.points[0].velocities.resize(3);
  const Eigen::Affine3d origin_to_base_out = robot_->GetRobotTransform();
  const Eigen::Vector3d origin_to_base_rpy = origin_to_base_out.rotation().eulerAngles(0, 1, 2);
  base_trajectory.points[0].positions[0] = origin_to_base_out.translation()[0];
  base_trajectory.points[0].positions[1] = origin_to_base_out.translation()[1];
  base_trajectory.points[0].positions[2] = origin_to_base_rpy[2];
  base_trajectory.points[0].velocities[0] = 0.0;
  base_trajectory.points[0].velocities[1] = 0.0;
  base_trajectory.points[0].velocities[2] = 0.0;
  base_trajectory.points[0].time_from_start = rclcpp::Duration::from_seconds(duration);

  // publish
  arm_trajectory.header.stamp = stamp;
  base_trajectory.header.stamp = stamp;
  for (const auto& publisher : arm_command_pubs_) {
    publisher->Publish(arm_trajectory, *latest_joint_states_);
  }
  base_command_pub_->publish(base_trajectory);
}

// Calculate the state of the robot after moving
bool PseudoEndeffectorController::CalcNextState(
    const tmc_manipulation_types::BaseMovementType& base_type,
    const Eigen::Affine3d& origin_to_next_end) {
  tmc_robot_kinematics_model::IKRequest ik_request(base_type);
  ik_request.use_joints = use_joints_;
  ik_request.frame_to_end = Eigen::Affine3d::Identity();
  ik_request.frame_name = endeffector_frame_name_;
  ik_request.origin_to_base = robot_->GetRobotTransform();
  ik_request.initial_angle = robot_->GetNamedAngle(use_joints_);
  ik_request.ref_origin_to_end = origin_to_next_end;
  if (base_type == tmc_manipulation_types::kRotationZ) {
    ik_request.weight.resize(use_joints_.size() + 1);
    ik_request.weight.head(use_joints_.size()) = ik_arm_weights_;
    ik_request.weight.tail(1) = ik_base_weights_.tail(1);
  } else if (base_type == tmc_manipulation_types::kPlanar) {
    ik_request.weight.resize(use_joints_.size() + 3);
    ik_request.weight.head(use_joints_.size()) = ik_arm_weights_;
    ik_request.weight.tail(3) = ik_base_weights_;
  }

  tmc_manipulation_types::JointState joint_state_out;
  Eigen::Affine3d origin_to_end_out;
  const auto result = ik_solver_->Solve(ik_request, joint_state_out, origin_to_end_out);
  if (result != tmc_robot_kinematics_model::kSuccess) {
    return false;
  }
  return true;
}

// Command_velocity -based callback common processing
void PseudoEndeffectorController::VelocityCallback(
    const geometry_msgs::msg::TwistStamped::SharedPtr& command,
    const tmc_manipulation_types::BaseMovementType& base_type) {
  const auto current_stamp = now();
  const auto diff_period = (current_stamp - last_command_stamp_).seconds();
  if (publish_period_.has_value() && diff_period < publish_period_.value()) {
    return;
  }

  if (!open_loop_control_) {
    if (!UpdateJointState()) {
      PublishIsSuccess(false);
      return;
    }
  }

  if (diff_period > discontinuous_period_ || command->header.frame_id != last_command_frame_) {
    // Processing as a new command => Processing from the current value
    if (open_loop_control_) {
      if (!UpdateJointState()) {
        PublishIsSuccess(false);
        return;
      }
    }
    origin_to_end_ = robot_->GetObjectTransform(endeffector_frame_name_);
    try {
      origin_to_base_ = robot_->GetObjectTransform(command->header.frame_id);
    } catch(std::domain_error& err) {
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 5 * 1000, "Cannot find frame: %s", command->header.frame_id.c_str());
      PublishIsSuccess(false);
      return;
    }
  } else {
    // Process as a continuous command value => Calculate the ideal movement and process it
    Eigen::Affine3d desired_transform;
    TwistToTransform(last_command_value_, diff_period, desired_transform);
    CalcOriginToNextEnd(origin_to_end_, origin_to_base_, desired_transform, origin_to_end_);
  }
  last_command_frame_ = command->header.frame_id;
  last_command_value_ = command->twist;

  Eigen::Affine3d transform;
  TwistToTransform(command->twist, velocity_duration_, transform);
  Eigen::Affine3d origin_to_next_end;
  CalcOriginToNextEnd(origin_to_end_, origin_to_base_, transform, origin_to_next_end);

  const auto result = CalcNextState(base_type, origin_to_next_end);
  if (result) {
    PublishJointCommand(now(), velocity_duration_);
    last_command_stamp_ = current_stamp;
  } else {
    RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 5 * 1000, "IK failed");
  }
  PublishIsSuccess(result);
}

// Command_velocity callback
void PseudoEndeffectorController::CommandVelocityCallback(
    const geometry_msgs::msg::TwistStamped::SharedPtr command) {
  VelocityCallback(command, tmc_manipulation_types::kRotationZ);
}

// Command_velocity_with_base callback
void PseudoEndeffectorController::CommandVelocityWithBaseCallback(
    const geometry_msgs::msg::TwistStamped::SharedPtr command) {
  VelocityCallback(command, tmc_manipulation_types::kPlanar);
}

// Publish a success or fail
void PseudoEndeffectorController::PublishIsSuccess(bool result) {
  std_msgs::msg::Bool success;
  success.data = result;
  success_pub_->publish(success);
}

Eigen::VectorXd PseudoEndeffectorController::GetWeightParameter(const std::string& parameter_name, uint32_t dof) {
  const auto ik_weights_vec = tmc_utils::GetParameter<std::vector<double>>(this, parameter_name, {});
  Eigen::VectorXd ik_weights_eigen = Eigen::VectorXd::Ones(dof);
  if (ik_weights_vec.size() != dof) {
    return ik_weights_eigen;
  }
  for (auto x : ik_weights_vec) {
    if (x <= std::numeric_limits<double>::min()) {
      return ik_weights_eigen;
    }
  }
  ik_weights_eigen = Eigen::Map<const Eigen::VectorXd>(&ik_weights_vec[0], ik_weights_vec.size());
  return ik_weights_eigen;
}

}  // namespace hsrb_pseudo_endeffector_controller
