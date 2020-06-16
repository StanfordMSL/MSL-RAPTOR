cmake_minimum_required(VERSION 3.0)
project(gtsam_test CXX)

set(CMAKE_VERBOSE_MAKEFILE 1)

# Find GTSAM, either from a local build, or from a Debian/Ubuntu package.
find_package(GTSAM REQUIRED)
find_package(rosbag REQUIRED)
find_package(std_msgs REQUIRED)

# message("Includes:" ${rosbag_INCLUDE_DIRS} )
# message("Libraries:" ${rosbag_LIBRARIES} )

include_directories( ${rosbag_INCLUDE_DIRS} ${std_msgs_INCLUDE_DIRS} )

add_executable(gtsam_test
  src/gtsam_test.cpp
)

# By using CMake exported targets, a simple "link" dependency introduces the
# include directories (-I) flags, links against Boost, and add any other
# required build flags (e.g. C++11, etc.)
target_link_libraries(gtsam_test PUBLIC gtsam ${rosbag_LIBRARIES} ${std_msgs_LIBRARIES} )
