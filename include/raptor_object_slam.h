#pragma once

// GTSAM Includes
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
// #include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearEquality.h> 
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <gtsam/nonlinear/utilities.h>

// ROS Includes
#include "ros/ros.h"
#include "std_msgs/String.h"
// #include <rosbag/bag.h>

// Misc Standard Includes
#include <math.h>
#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>
#include <sstream>

// Custom Includes
#include "file_io_utils.h"

// Type Defs
typedef vector<tuple<double, gtsam::Pose3>> object_data_vec_t;  // vector of tuples(double, pose message)
typedef tuple<double, int, gtsam::Pose3, gtsam::Pose3> data_tuple; // double, int (id), pose gt, pose est
typedef vector<data_tuple> object_est_gt_data_vec_t; //vector of tuples(double, int (id), pose gt, pose est)

// Structs
struct obj_param_t {
  string long_name;
  string short_name;
  int obj_id;
  bool b_rm_roll;
  bool b_rm_pitch;
  bool b_rm_yaw;
  obj_param_t() { // default values
    long_name = ""; short_name = ""; obj_id = -1;
    b_rm_roll = false; b_rm_pitch = false; b_rm_yaw = false;
  }
  obj_param_t(string long_name_, string short_name_, int obj_id_) {
    obj_param_t();
    long_name = long_name_; short_name = short_name_; obj_id = obj_id_;
  }
  obj_param_t(string long_name_, string short_name_, int obj_id_, bool rm_r_, bool rm_p_, bool rm_y_) {
    obj_param_t();
    long_name = long_name_; short_name = short_name_; obj_id = obj_id_;
    b_rm_roll = rm_r_; b_rm_pitch = rm_p_; b_rm_yaw = rm_y_;
  }
};