#pragma once

#include <gtsam/geometry/Pose3.h>

// Type Defs
typedef vector<tuple<double, gtsam::Pose3>> object_data_vec_t;  // vector of tuples(double, pose message)
typedef tuple<double, int, gtsam::Pose3, gtsam::Pose3> data_tuple; // double, int (id), pose gt, pose est
typedef vector<data_tuple> object_est_gt_data_vec_t; //vector of tuples(double, int (id), pose gt, pose est)

// Structs
struct obj_param_t {
  string instance_name;  // name of the specific object e.g. quad7 or benjamin
  string class_name;     // name of the class e.g. mslquad or person
  int obj_id;
  bool b_rm_roll;
  bool b_rm_pitch;
  bool b_rm_yaw;
  obj_param_t() { // default values
    instance_name = ""; class_name = ""; obj_id = -1;
    b_rm_roll = false; b_rm_pitch = false; b_rm_yaw = false;
  }
  obj_param_t(string instance_name_, string class_name_, int obj_id_) {
    obj_param_t();
    instance_name = instance_name_; class_name = class_name_; obj_id = obj_id_;
  }
  obj_param_t(string instance_name_, string class_name_, int obj_id_, bool rm_r_, bool rm_p_, bool rm_y_) {
    obj_param_t();
    instance_name = instance_name_; class_name = class_name_; obj_id = obj_id_;
    b_rm_roll = rm_r_; b_rm_pitch = rm_p_; b_rm_yaw = rm_y_;
  }
};