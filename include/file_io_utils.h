#pragma once

// GTSAM Includes
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>

// ROS Includes
#include <rosbag/bag.h>
#include <rosbag/view.h>
// #include <std_msgs/Int32.h>
#include <geometry_msgs/Pose.h> 
#include <geometry_msgs/PoseStamped.h> 
// #include <sensor_msgs/CameraInfo.h> // geometry_msgs/PoseStamped sensor_msgs/CameraInfo sensor_msgs/Image tf/tfMessage 

#include <msl_raptor/AngledBbox.h>
#include <msl_raptor/AngledBbox.h>
#include <msl_raptor/TrackedObject.h>
#include <msl_raptor/TrackedObjects.h>

#include "math_utils.h"
#include "shared_imports.h"

namespace rslam_utils {
    void load_rosbag(set<double> &times, std::map<std::string, object_est_gt_data_vec_t> obj_data, 
                    std::string rosbag_fn, std::string ego_ns, std::map<std::string, obj_param_t> obj_param_map, double dt_thresh);
    // void load_raptor_output_rosbag(set<double> &times, object_data_vec_t &ego_data_est, std::map<std::string, object_data_vec_t> &ado_data_est, 
    //                                 std::string rosbag_fn, std::string ego_ns, std::map<std::string, obj_param_t> obj_param_map);
    // void load_gt_rosbag(set<double> &times, object_data_vec_t &ego_data_gt, std::map<std::string, object_data_vec_t> &ado_data_gt, 
    //                         std::string rosbag_fn, std::string ego_ns, std::map<std::string, obj_param_t> obj_param_map, double dt_thresh);
    gtsam::Pose3 ros_geo_pose_to_gtsam(geometry_msgs::Pose ros_pose);

    void read_gt_datafiles(const string fn, std::map<double, pair<gtsam::Pose3, gtsam::Pose3> >& time_tf_w_ego_map, set<double> &times);
    void read_data_from_one_log(const string fn, object_data_vec_t& obj_data, set<double> &times);
    void sync_est_and_gt(object_data_vec_t data_est, object_data_vec_t data_gt, object_est_gt_data_vec_t& data, obj_param_t params, double dt_thresh);
    void write_results_csv(string fn, std::map<gtsam::Symbol, double> ego_time_map, std::map<gtsam::Symbol, gtsam::Pose3> tf_w_gt_map, std::map<gtsam::Symbol, gtsam::Pose3> tf_w_est_preslam_map, std::map<gtsam::Symbol, gtsam::Pose3> tf_w_est_postslam_map, std::map<gtsam::Symbol, std::map<gtsam::Symbol, pair<gtsam::Pose3, gtsam::Pose3> > > tf_ego_ado_maps);
    void write_all_traj_csv(string fn, std::map<gtsam::Symbol, std::map<double, pair<gtsam::Pose3, gtsam::Pose3> > > & all_trajs);
    string pose_to_string_line(gtsam::Pose3 p);
}