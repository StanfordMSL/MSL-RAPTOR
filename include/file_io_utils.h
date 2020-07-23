#pragma once

// GTSAM Includes
#include <gtsam/geometry/Pose3.h>

// ROS Includes
#include <rosbag/bag.h>
#include <rosbag/view.h>
// #include <std_msgs/Int32.h>
#include <geometry_msgs/PoseStamped.h> 
// #include <sensor_msgs/CameraInfo.h> // geometry_msgs/PoseStamped sensor_msgs/CameraInfo sensor_msgs/Image tf/tfMessage 

#include <msl_raptor/AngledBbox.h>
#include <msl_raptor/AngledBbox.h>
#include <msl_raptor/TrackedObject.h>
#include <msl_raptor/TrackedObjects.h>

#include "shared_imports.h"

namespace rslam_utils {
    void load_raptor_output_rosbag(std::string rosbag_fn);
    void load_gt_rosbag(std::string, std::string ego_ns, std::map<std::string, obj_param_t> obj_param_map);
    gtsam::Pose3 ros_geo_pose_to_gtsam_pose3(geometry_msgs::Pose ros_pose);
}