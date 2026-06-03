cmake_minimum_required(VERSION 3.8)
project(fetchpush_action)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(fetchpush_action_interface REQUIRED)
find_package(fetchpush_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)

set(CMAKE_PREFIX_PATH /home/gianluca/libtorch)
find_package(Torch REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(fetchpush_server src/fetchpush_server.cpp)
target_include_directories(fetchpush_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(fetchpush_server "${TORCH_LIBRARIES}")
ament_target_dependencies(fetchpush_server
  rclcpp
  rclcpp_action
  rclcpp_components
  fetchpush_action_interface
  fetchpush_msgs)

install(TARGETS fetchpush_server
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})
  
ament_package()
