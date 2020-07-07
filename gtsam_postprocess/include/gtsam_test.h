#pragma once
/**
 * @file gtsam_test.cpp
 * @brief Attempt to read in data and process with isam
 * @author Adam Caccavale
 */

/**
 * Attempt to read in a data produced by MSL-RAPTOR and further process it with gtsam
 */
#include <gtsam/nonlinear/utilities.h>

using namespace std;
using namespace gtsam;



// GTSAM-RAPTOR Function
void run_isam(const set<double> &times, const object_est_gt_data_vec_t& obj_data, const map<std::string, obj_param_t> &obj_params_map, double dt_thresh);
void run_batch_slam(const set<double> &times, const object_est_gt_data_vec_t& obj_data, const map<std::string, obj_param_t> &obj_params_map, double dt_thresh);
void simulate_measurement_from_bag(double ego_time, 
                                  int t_ind, 
                                  int &obj_list_ind, 
                                  const object_est_gt_data_vec_t& obj_data, 
                                  const map<std::string, obj_param_t> &obj_params_map, 
                                  double dt_thresh,
                                  map<Symbol, Pose3> &tf_w_gt_t0_map,
                                  map<Symbol, Pose3> &tf_w_est_t0_map,
                                  Pose3 &tf_w_ego_gt,      // output
                                  Pose3 &tf_w_ego_est,     // output
                                  vector<raptor_measurement_t> &raptor_measurement_vec);


void debug_func();
float fix_it(float f);

