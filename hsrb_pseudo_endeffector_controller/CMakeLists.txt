cmake_minimum_required(VERSION 3.5)
project(hsrb_pseudo_endeffector_controller)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tmc_manipulation_types REQUIRED)
find_package(tmc_manipulation_util REQUIRED)
find_package(tmc_robot_kinematics_model REQUIRED)
find_package(tmc_utils REQUIRED)
find_package(trajectory_msgs REQUIRED)

find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME}_lib src/${PROJECT_NAME}.cpp)
set_property(TARGET ${PROJECT_NAME}_lib PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(${PROJECT_NAME}_lib PUBLIC ${EIGEN3_INCLUDE_DIRS})
ament_target_dependencies(${PROJECT_NAME}_lib geometry_msgs nav_msgs rclcpp sensor_msgs std_msgs tmc_manipulation_types tmc_manipulation_util tmc_robot_kinematics_model tmc_utils trajectory_msgs)

add_executable(${PROJECT_NAME} src/node.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib)

install(TARGETS ${PROJECT_NAME}_lib ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(tmc_manipulation_tests REQUIRED)

  ament_add_gtest(hsrb_pseudo_endeffector_controller_test test/hsrb_pseudo_endeffector_controller-test.cpp)
  target_link_libraries(hsrb_pseudo_endeffector_controller_test ${PROJECT_NAME}_lib)
  ament_target_dependencies(hsrb_pseudo_endeffector_controller_test tmc_manipulation_tests)
endif()

ament_export_dependencies(geometry_msgs nav_msgs rclcpp sensor_msgs std_msgs tmc_manipulation_types tmc_manipulation_util tmc_robot_kinematics_model trajectory_msgs)
ament_package()
