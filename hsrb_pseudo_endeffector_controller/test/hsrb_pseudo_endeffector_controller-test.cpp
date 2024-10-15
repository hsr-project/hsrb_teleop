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
#include <functional>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>

#include <tmc_manipulation_tests/configs.hpp>

#include "../src/hsrb_pseudo_endeffector_controller.hpp"

namespace {
constexpr double kWaitForMessageTimeout = 0.3;
constexpr double kPollingRate = 100.0;
const char* const kServerNodeName = "pseudo_endeffector_controller_node/";
constexpr double kEpsilon = 1e-3;

template <class MessageType>
class TestSubscriber {
 public:
  using Ptr = std::shared_ptr<TestSubscriber>;

  TestSubscriber(const rclcpp::Node::SharedPtr& node, const std::string& topic_name)
      : is_sub_(false), clock_(node->get_clock()), count_(0) {
    auto callback = [this](const std::shared_ptr<MessageType> msg) -> void {
      this->value_ = *msg;
      this->is_sub_ = true;
      ++this->count_;
    };
    subscriber_ = node->create_subscription<MessageType>(topic_name, 1, callback);
  }

  void Clear() {
    is_sub_ = false;
    count_ = 0;
  }
  bool WaitForMessage(std::function<void()> spin_some_func, MessageType& dst_value) {
    const auto timeout = clock_->now() + rclcpp::Duration::from_seconds(kWaitForMessageTimeout);
    rclcpp::WallRate rate(kPollingRate);
    while (rclcpp::ok()) {
      if (clock_->now() > timeout) {
        return false;
      }
      if (is_sub_) {
        dst_value = value_;
        return true;
      }
      spin_some_func();
      rate.sleep();
    }
    return false;
  }

  uint32_t count() const { return count_; }

 private:
  typename rclcpp::Subscription<MessageType>::SharedPtr subscriber_;
  MessageType value_;
  bool is_sub_;
  rclcpp::Clock::SharedPtr clock_;
  uint32_t count_;
};


class RobotStatePublisher {
 public:
  using Ptr = std::shared_ptr<RobotStatePublisher>;

  explicit RobotStatePublisher(const rclcpp::Node::SharedPtr& node) {
    joint_states_pub_ = node->create_publisher<sensor_msgs::msg::JointState>("joint_states", 1);
    odom_pub_ = node->create_publisher<nav_msgs::msg::Odometry>("odom", 1);
  }

  void ResetRobotState(std::function<void()> spin_some_func) {
    sensor_msgs::msg::JointState joint;
    joint.name = {"arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"};
    joint.position = {0.0, 0.0, 0.0, -M_PI / 2.0, 0.0};
    joint_states_pub_->publish(joint);

    PublishOdom(0.0);

    spin_some_func();
  }

  void PublishOdom(double odom_x) {
    nav_msgs::msg::Odometry odom;
    odom.pose.pose.position.x = odom_x;
    odom.pose.pose.orientation.w = 1.0;
    odom_pub_->publish(odom);
  }

 private:
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
};

}  // anonymous namespace

namespace hsrb_pseudo_endeffector_controller {

class PseudoEndeffectorControllerTest : public ::testing::Test {
 public:
  void SpinSome() {
    if (client_node_) rclcpp::spin_some(client_node_);
    if (server_node_) rclcpp::spin_some(server_node_);
  }

 protected:
  void SetUp() override;

  bool IsSuccess();
  void ValidateFailure();
  std::map<std::string, double> ReceiveCommandTrajectory();

  void ClearSubscribers();

  rclcpp::Node::SharedPtr client_node_;
  rclcpp::Node::SharedPtr server_node_;

  rclcpp::NodeOptions default_options_;

  TestSubscriber<trajectory_msgs::msg::JointTrajectory>::Ptr arm_command_sub_;
  TestSubscriber<trajectory_msgs::msg::JointTrajectory>::Ptr base_command_sub_;
  TestSubscriber<std_msgs::msg::Bool>::Ptr success_sub_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr command_velocity_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr command_velocity_with_base_pub_;

  RobotStatePublisher::Ptr robot_state_pub_;
};

void PseudoEndeffectorControllerTest::SetUp() {
  client_node_ = rclcpp::Node::make_shared("client");
  default_options_.parameter_overrides() = {
      rclcpp::Parameter("robot_description", tmc_manipulation_tests::hsrb::GetUrdf()),
      rclcpp::Parameter("arm_trajectory_controller.joints",
                        std::vector<std::string>{"arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
                                                 "wrist_flex_joint", "wrist_roll_joint"}),
      rclcpp::Parameter("publish_rate", 0.0)};

  arm_command_sub_ = std::make_shared<TestSubscriber<trajectory_msgs::msg::JointTrajectory>>(
      client_node_, "arm_trajectory_controller/joint_trajectory");
  base_command_sub_ = std::make_shared<TestSubscriber<trajectory_msgs::msg::JointTrajectory>>(
      client_node_, "omni_base_controller/joint_trajectory");
  success_sub_ = std::make_shared<TestSubscriber<std_msgs::msg::Bool>>(
      client_node_, std::string(kServerNodeName) + "success");

  command_velocity_pub_ = client_node_->create_publisher<geometry_msgs::msg::TwistStamped>(
      std::string(kServerNodeName) + "command_velocity", 1);
  command_velocity_with_base_pub_ = client_node_->create_publisher<geometry_msgs::msg::TwistStamped>(
      std::string(kServerNodeName) + "command_velocity_with_base", 1);

  robot_state_pub_ = std::make_shared<RobotStatePublisher>(client_node_);
}

bool PseudoEndeffectorControllerTest::IsSuccess() {
  std_msgs::msg::Bool success;
  EXPECT_TRUE(success_sub_->WaitForMessage(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this), success));
  return success.data;
}

void PseudoEndeffectorControllerTest::ValidateFailure() {
  EXPECT_FALSE(IsSuccess());
  auto spin_func = std::bind(&PseudoEndeffectorControllerTest::SpinSome, this);
  trajectory_msgs::msg::JointTrajectory command;
  EXPECT_FALSE(arm_command_sub_->WaitForMessage(spin_func, command));
  EXPECT_FALSE(base_command_sub_->WaitForMessage(spin_func, command));
}

std::map<std::string, double> PseudoEndeffectorControllerTest::ReceiveCommandTrajectory() {
  trajectory_msgs::msg::JointTrajectory arm_trajectory;
  auto spin_func = std::bind(&PseudoEndeffectorControllerTest::SpinSome, this);
  EXPECT_TRUE(arm_command_sub_->WaitForMessage(spin_func, arm_trajectory));
  EXPECT_EQ(arm_trajectory.joint_names.size(), 5);
  EXPECT_EQ(arm_trajectory.points.size(), 1);
  EXPECT_EQ(arm_trajectory.joint_names.size(), arm_trajectory.points[0].positions.size());
  std::map<std::string, double> result;
  for (int i = 0; i < arm_trajectory.joint_names.size(); ++i) {
    result[arm_trajectory.joint_names[i]] = arm_trajectory.points[0].positions[i];
  }

  trajectory_msgs::msg::JointTrajectory base_trajectory;
  EXPECT_TRUE(base_command_sub_->WaitForMessage(spin_func, base_trajectory));
  EXPECT_EQ(base_trajectory.joint_names.size(), 3);
  EXPECT_EQ(base_trajectory.points.size(), 1);
  EXPECT_EQ(base_trajectory.joint_names.size(), base_trajectory.points[0].positions.size());
  for (int i = 0; i < base_trajectory.joint_names.size(); ++i) {
    result[base_trajectory.joint_names[i]] = base_trajectory.points[0].positions[i];
  }

  EXPECT_EQ(arm_trajectory.header.stamp, base_trajectory.header.stamp);
  EXPECT_EQ(arm_trajectory.points[0].time_from_start, base_trajectory.points[0].time_from_start);
  result["time_from_start"] = rclcpp::Duration(arm_trajectory.points[0].time_from_start).seconds();

  return result;
}

void PseudoEndeffectorControllerTest::ClearSubscribers() {
  arm_command_sub_->Clear();
  base_command_sub_->Clear();
  success_sub_->Clear();
}


// Test of issuance cycle restriction
TEST_F(PseudoEndeffectorControllerTest, PublishRate) {
  bool has_param = false;
  for (auto i = 0; i < default_options_.parameter_overrides().size(); ++i) {
    if (default_options_.parameter_overrides()[i].get_name() == "publish_rate") {
      has_param = true;
      default_options_.parameter_overrides()[i] = rclcpp::Parameter("publish_rate", 40.0);
    }
  }
  if (!has_param) {
    default_options_.parameter_overrides().push_back(rclcpp::Parameter("publish_rate", 40.0));
  }

  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "hand_palm_link";
  twist.twist.linear.x = 0.001;

  {
    const auto timeout = client_node_->now() + rclcpp::Duration::from_seconds(0.5);
    while (client_node_->now() < timeout) {
      command_velocity_pub_->publish(twist);
      rclcpp::spin_some(server_node_);
      rclcpp::spin_some(client_node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  // The ideal is only 10 because it is only issued, but it may not be 10 at the timing of PUB/SUB, about 8 should come out stably.
  EXPECT_GT(arm_command_sub_->count(), 8);
  EXPECT_LE(arm_command_sub_->count(), 10);

  arm_command_sub_->Clear();
  {
    const auto timeout = client_node_->now() + rclcpp::Duration::from_seconds(0.5);
    while (client_node_->now() < timeout) {
      command_velocity_pub_->publish(twist);
      rclcpp::spin_some(server_node_);
      rclcpp::spin_some(client_node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  // Since it is limited to 40Hz, the ideal is 20, but it should be less than 20, close to 20.
  EXPECT_GT(arm_command_sub_->count(), 15);
  EXPECT_LE(arm_command_sub_->count(), 20);
}

// /Test when there is no Robot_description
TEST_F(PseudoEndeffectorControllerTest, NoRobotDescription) {
  EXPECT_ANY_THROW(server_node_ = std::make_shared<PseudoEndeffectorController>());
}

// If Joints_states, etc., do not issue Command
TEST_F(PseudoEndeffectorControllerTest, NoRobotState) {
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "hand_palm_link";
  command_velocity_pub_->publish(twist);

  ValidateFailure();
}

// A frame that does not exist
TEST_F(PseudoEndeffectorControllerTest, InvalidFrameName) {
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "nolink";
  command_velocity_pub_->publish(twist);

  ValidateFailure();
}

// Testing with Hand_palm_link is the standard frame without a bogie movement
TEST_F(PseudoEndeffectorControllerTest, CommandVelocityOnHandFrame) {
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "hand_palm_link";
  twist.twist.linear.x = 0.1;
  twist.twist.angular.z = -0.1;

  rclcpp::WallRate rate(rclcpp::Duration::from_seconds(0.25).to_chrono<std::chrono::nanoseconds>());
  command_velocity_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto first_command = ReceiveCommandTrajectory();
  // Posture after 0.5 seconds
  // The other axis also moves slightly, so make the expected value from the output value.
  // Check if 0.05 in the vertical direction (mainly lift)
  // Check if the wrist ROLL axis is rotating -0.05
  EXPECT_DOUBLE_EQ(first_command.at("time_from_start"), 0.5);
  EXPECT_NEAR(first_command.at("odom_x"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_y"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_t"), 0.0025, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_lift_joint"), 0.1 * 0.5, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_flex_joint"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_roll_joint"), -0.002, kEpsilon);
  EXPECT_NEAR(first_command.at("wrist_flex_joint"), -M_PI / 2.0, kEpsilon);
  EXPECT_NEAR(first_command.at("wrist_roll_joint"), -0.1 * 0.5, kEpsilon);

  ClearSubscribers();
  rate.sleep();

  command_velocity_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto second_command = ReceiveCommandTrajectory();
  // Posture after 0.75 seconds
  // Similarly, check the vertical direction and wrist ROLL axis.
  EXPECT_DOUBLE_EQ(second_command.at("time_from_start"), 0.5);
  EXPECT_NEAR(second_command.at("odom_x"), 0.0, kEpsilon);
  EXPECT_NEAR(second_command.at("odom_y"), 0.0, kEpsilon);
  EXPECT_NEAR(second_command.at("odom_t"), 0.0058, kEpsilon);
  EXPECT_NEAR(second_command.at("arm_lift_joint"), 0.1 * 0.75, kEpsilon);
  EXPECT_NEAR(second_command.at("arm_flex_joint"), -0.001, kEpsilon);
  EXPECT_NEAR(second_command.at("arm_roll_joint"), -0.006, kEpsilon);
  EXPECT_NEAR(second_command.at("wrist_flex_joint"), -1.569, kEpsilon);
  EXPECT_NEAR(second_command.at("wrist_roll_joint"), -0.1 * 0.75, kEpsilon);
}

// Even if the command value is sent continuously, solve the IK using the current value
TEST_F(PseudoEndeffectorControllerTest, CloseLoopControl) {
  default_options_.parameter_overrides().push_back(rclcpp::Parameter("open_loop_control", false));
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "base_footprint";
  twist.twist.linear.x = 0.1;

  rclcpp::WallRate rate(rclcpp::Duration::from_seconds(0.25).to_chrono<std::chrono::nanoseconds>());
  command_velocity_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto first_command = ReceiveCommandTrajectory();
  // The posture after 0.5 seconds, raising the rise axis, the trolley does not move.
  EXPECT_DOUBLE_EQ(first_command.at("time_from_start"), 0.5);
  EXPECT_NEAR(first_command.at("odom_x"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_y"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_t"), 0.0, kEpsilon);
  EXPECT_GT(first_command.at("arm_lift_joint"), 0.0);
  EXPECT_LT(first_command.at("arm_flex_joint"), 0.0);
  EXPECT_NEAR(first_command.at("arm_roll_joint"), 0.0, kEpsilon);
  EXPECT_GT(first_command.at("wrist_flex_joint"), -M_PI / 2.0);
  EXPECT_NEAR(first_command.at("wrist_roll_joint"), 0.0, kEpsilon);

  robot_state_pub_->PublishOdom(0.1 * 0.75);

  ClearSubscribers();
  rate.sleep();
  SpinSome();
  command_velocity_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto second_command = ReceiveCommandTrajectory();
  // The posture after 0.75 seconds, the movement of the bogie will be feedback, so the arm is almost moved.
  EXPECT_DOUBLE_EQ(second_command.at("time_from_start"), 0.5);
  EXPECT_NEAR(second_command.at("odom_x"), 0.1 * 0.75, kEpsilon);
  EXPECT_NEAR(second_command.at("odom_y"), 0.0, kEpsilon);
  EXPECT_NEAR(second_command.at("odom_t"), 0.0, kEpsilon);
  EXPECT_NEAR(second_command.at("arm_lift_joint"), 0.0, kEpsilon);
  EXPECT_NEAR(second_command.at("arm_flex_joint"), 0.0, 0.01);
  EXPECT_NEAR(second_command.at("arm_roll_joint"), 0.0, kEpsilon);
  EXPECT_NEAR(second_command.at("wrist_flex_joint"), -M_PI / 2.0, 0.01);
  EXPECT_NEAR(second_command.at("wrist_roll_joint"), 0.0, kEpsilon);
}

// If the command value is continuously sent, solve the IK without feedback on the present value
TEST_F(PseudoEndeffectorControllerTest, OpenLoopControl) {
  default_options_.parameter_overrides().push_back(rclcpp::Parameter("open_loop_control", true));
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "base_footprint";
  twist.twist.linear.x = 0.1;

  rclcpp::WallRate rate(rclcpp::Duration::from_seconds(0.25).to_chrono<std::chrono::nanoseconds>());
  command_velocity_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto first_command = ReceiveCommandTrajectory();
  // The posture after 0.5 seconds, the rising axis should be raised, and the hands should be put forward, but the bogie does not move because it is a setting without a trolley.
  EXPECT_DOUBLE_EQ(first_command.at("time_from_start"), 0.5);
  EXPECT_NEAR(first_command.at("odom_x"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_y"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_t"), 0.0, kEpsilon);
  EXPECT_GT(first_command.at("arm_lift_joint"), 0.0);
  EXPECT_LT(first_command.at("arm_flex_joint"), 0.0);
  EXPECT_NEAR(first_command.at("arm_roll_joint"), 0.0, kEpsilon);
  EXPECT_GT(first_command.at("wrist_flex_joint"), -M_PI / 2.0);
  EXPECT_NEAR(first_command.at("wrist_roll_joint"), 0.0, kEpsilon);

  robot_state_pub_->PublishOdom(0.1 * 0.75);

  ClearSubscribers();
  rate.sleep();
  SpinSome();
  command_velocity_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto second_command = ReceiveCommandTrajectory();
  // Posture after 0.75 seconds, the movement of the bogie is not feedback+no trolle has been set, so put out more hands.
  EXPECT_DOUBLE_EQ(second_command.at("time_from_start"), 0.5);
  EXPECT_NEAR(second_command.at("odom_x"), 0.0, kEpsilon);
  EXPECT_NEAR(second_command.at("odom_y"), 0.0, kEpsilon);
  EXPECT_NEAR(second_command.at("odom_t"), 0.0, kEpsilon);
  EXPECT_GT(second_command.at("arm_lift_joint"), first_command.at("arm_lift_joint"));
  EXPECT_LT(second_command.at("arm_flex_joint"), first_command.at("arm_flex_joint"));
  EXPECT_NEAR(second_command.at("arm_roll_joint"), 0.0, kEpsilon);
  EXPECT_GT(second_command.at("wrist_flex_joint"), first_command.at("wrist_flex_joint"));
  EXPECT_NEAR(second_command.at("wrist_roll_joint"), 0.0, kEpsilon);
}

// Test with Base_footPrint for the bogie without movement
TEST_F(PseudoEndeffectorControllerTest, CommandVelocityOnFootprint) {
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "base_footprint";
  twist.twist.linear.z = 0.1;
  twist.twist.angular.x = -0.1;
  command_velocity_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto first_command = ReceiveCommandTrajectory();
  // The other axis also moves slightly, so make the expected value from the output value.
  // Check if 0.05 in the vertical direction (mainly lift)
  // Check if the wrist ROLL axis is rotating -0.05
  EXPECT_DOUBLE_EQ(first_command.at("time_from_start"), 0.5);
  EXPECT_NEAR(first_command.at("odom_x"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_y"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_t"), 0.0025, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_lift_joint"), 0.1 * 0.5, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_flex_joint"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_roll_joint"), -0.002, kEpsilon);
  EXPECT_NEAR(first_command.at("wrist_flex_joint"), -M_PI / 2.0, kEpsilon);
  EXPECT_NEAR(first_command.at("wrist_roll_joint"), -0.1 * 0.5, kEpsilon);
}

// Testing with trolleys has been tested with Hand_palm_link
TEST_F(PseudoEndeffectorControllerTest, CommandVelocityWithBaseOnHandFrame) {
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "hand_palm_link";
  twist.twist.linear.z = 0.1;
  command_velocity_with_base_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto first_command = ReceiveCommandTrajectory();
  // The bogie moves (the expected value is created from the answer)
  // Confirm that the bogie (ODOM_X) moves nearly 0.05
  EXPECT_DOUBLE_EQ(first_command.at("time_from_start"), 0.5);
  EXPECT_NEAR(first_command.at("odom_x"), 0.046, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_y"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_t"), -0.0018, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_lift_joint"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_flex_joint"), -0.0081, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_roll_joint"), 0.0018, kEpsilon);
  EXPECT_NEAR(first_command.at("wrist_flex_joint"), -M_PI / 2.0 + 0.0081, kEpsilon);
  EXPECT_NEAR(first_command.at("wrist_roll_joint"), 0.0, kEpsilon);
}

// Testing with bogie movement with Base_footPrint
TEST_F(PseudoEndeffectorControllerTest, CommandVelocityWithBaseOnFootprint) {
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "base_footprint";
  twist.twist.linear.x = 0.1;
  command_velocity_with_base_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto first_command = ReceiveCommandTrajectory();
  // The bogie moves (the expected value is created from the answer)
  // Confirm that the bogie (ODOM_X) moves nearly 0.05
  EXPECT_DOUBLE_EQ(first_command.at("time_from_start"), 0.5);
  EXPECT_NEAR(first_command.at("odom_x"), 0.046, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_y"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_t"), -0.0018, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_lift_joint"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_flex_joint"), -0.0081, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_roll_joint"), 0.0018, kEpsilon);
  EXPECT_NEAR(first_command.at("wrist_flex_joint"), -M_PI / 2.0 + 0.0081, kEpsilon);
  EXPECT_NEAR(first_command.at("wrist_roll_joint"), 0.0, kEpsilon);
}

// IK without trolleys has failed
TEST_F(PseudoEndeffectorControllerTest, CommandVelocityIKFailed) {
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "hand_palm_link";
  twist.twist.linear.x = 2.0;
  command_velocity_pub_->publish(twist);

  ValidateFailure();
}

// IK failed with trolleys
TEST_F(PseudoEndeffectorControllerTest, CommandVelocityWithBaseIKFailed) {
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "hand_palm_link";
  twist.twist.linear.x = 2.0;
  command_velocity_with_base_pub_->publish(twist);

  ValidateFailure();
}

// Test for IK_Delta = 1000, Velocity_duration = 10
TEST_F(PseudoEndeffectorControllerTest, WithParameter) {
  default_options_.parameter_overrides().push_back(rclcpp::Parameter("velocity_duration", 10.0));
  default_options_.parameter_overrides().push_back(rclcpp::Parameter("ik_delta", 1000.0));
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "hand_palm_link";
  twist.twist.linear.z = 0.01;
  command_velocity_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto first_command = ReceiveCommandTrajectory();
  // Confirm that IK_DELTA is large, so the state that does not move at all will be solved.
  // Confirm that the default value of Time_from_start becomes 10.0
  EXPECT_DOUBLE_EQ(first_command.at("time_from_start"), 10.0);
  EXPECT_NEAR(first_command.at("odom_x"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_y"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("odom_t"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_lift_joint"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_flex_joint"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("arm_roll_joint"), 0.0, kEpsilon);
  EXPECT_NEAR(first_command.at("wrist_flex_joint"), -M_PI / 2.0, kEpsilon);
  EXPECT_NEAR(first_command.at("wrist_roll_joint"), 0.0, kEpsilon);
}

// Testing whether the bogie weight is reflected
TEST_F(PseudoEndeffectorControllerTest, IkBaseWeights) {
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "base_footprint";
  twist.twist.linear.x = 0.1;
  command_velocity_with_base_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto default_command = ReceiveCommandTrajectory();

  ClearSubscribers();

  default_options_.parameter_overrides().push_back(
      rclcpp::Parameter("ik_base_weights", std::vector<double>({50.0, 1.0, 1.0})));
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  command_velocity_with_base_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto test_command = ReceiveCommandTrajectory();

  // Instead of not moving, you should put your hands forward with your shoulder Flex.
  EXPECT_LT(test_command.at("odom_x"), default_command.at("odom_x"));
  EXPECT_LT(test_command.at("arm_flex_joint"), default_command.at("arm_flex_joint"));
}

// Testing whether arm weight is reflected
TEST_F(PseudoEndeffectorControllerTest, IkArmWeights) {
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  geometry_msgs::msg::TwistStamped twist;
  twist.header.frame_id = "base_footprint";
  twist.twist.linear.x = 0.1;
  command_velocity_with_base_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto default_command = ReceiveCommandTrajectory();

  ClearSubscribers();

  default_options_.parameter_overrides().push_back(
      rclcpp::Parameter("ik_arm_weights", std::vector<double>({1.0, 50.0, 1.0, 1.0, 1.0})));
  server_node_ = std::make_shared<PseudoEndeffectorController>(default_options_);
  robot_state_pub_->ResetRobotState(std::bind(&PseudoEndeffectorControllerTest::SpinSome, this));

  command_velocity_with_base_pub_->publish(twist);

  ASSERT_TRUE(IsSuccess());
  const auto test_command = ReceiveCommandTrajectory();

  // Since the shoulder Flex does not move, it should move on the bogie
  EXPECT_GT(test_command.at("odom_x"), default_command.at("odom_x"));
  EXPECT_GT(test_command.at("arm_flex_joint"), default_command.at("arm_flex_joint"));
}


}  // namespace hsrb_pseudo_endeffector_controller


int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
