cmake_minimum_required(VERSION 3.0)
project(gtsam_test CXX)

set(CMAKE_VERBOSE_MAKEFILE 0)

# Set for debugging
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE RelWithDebInfo)

# Find GTSAM, either from a local build, or from a Debian/Ubuntu package.
find_package(GTSAM REQUIRED)

add_executable(gtsam_test src/gtsam_test.cpp)

# By using CMake exported targets, a simple "link" dependency introduces the
# include directories (-I) flags, links against Boost, and add any other
# required build flags (e.g. C++11, etc.)
target_link_libraries(gtsam_test PUBLIC 
    gtsam 
)

# # This clears the terminal when make is called
# add_custom_target(clearing_terminal ALL
#     COMMAND clear
#     DEPENDS ${gtsam_test}
# )