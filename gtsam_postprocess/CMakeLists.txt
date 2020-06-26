cmake_minimum_required(VERSION 3.0)
project(gtsam_test CXX)

set(CMAKE_VERBOSE_MAKEFILE 0)

# Set for debugging
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE RelWithDebInfo)

# Find GTSAM, either from a local build, or from a Debian/Ubuntu package.
find_package(GTSAM REQUIRED)
find_package(rosbag REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf REQUIRED)

# message("Includes:" ${rosbag_INCLUDE_DIRS} )
# message("Libraries:" ${rosbag_LIBRARIES} )

include_directories( 
    ${rosbag_INCLUDE_DIRS} 
    ${std_msgs_INCLUDE_DIRS} 
    ${geometry_msgs_INCLUDE_DIRS} 
    ${sensor_msgs_INCLUDE_DIRS}
    ${tf_INCLUDE_DIRS}
)


# custom command to clear terminal when "make" is called
# add_custom_target(tmp clear)

add_executable(gtsam_test src/gtsam_test.cpp)

# By using CMake exported targets, a simple "link" dependency introduces the
# include directories (-I) flags, links against Boost, and add any other
# required build flags (e.g. C++11, etc.)
target_link_libraries(gtsam_test PUBLIC 
    gtsam 
    ${rosbag_LIBRARIES} 
    ${std_msgs_LIBRARIES} 
    ${geometry_msgs_LIBRARIES} 
    ${sensor_msgs_LIBRARIES} 
    ${tf_LIBRARIES} 
)

# # This clears the terminal when cmake is called
# execute_process (COMMAND bash -c "clear")

# This clears the terminal when cmake is called
add_custom_target(clearing_terminal ALL
    COMMAND clear
    DEPENDS ${gtsam_test}
)