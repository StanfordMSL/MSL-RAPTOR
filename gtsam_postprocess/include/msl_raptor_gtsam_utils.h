#pragma once
/**
 * @file msl_raptor_gtsam_utils.h
 * @brief helper functions for raptor + gtsam
 * @author Adam Caccavale
 */

/**
 * helper functions for raptor + gtsam
 */

// // We will use Pose3 variables (x, y, z, Rot3) to represent the robot/landmark positions
// #include <gtsam/geometry/Pose3.h>

// // Once the optimized values have been calculated, we can also calculate the marginal covariance
// // of desired variables
// #include <gtsam/nonlinear/Marginals.h>

// // The nonlinear solvers within GTSAM are iterative solvers, meaning they linearize the
// // nonlinear functions around an initial linearization point, then solve the linear system
// // to update the linearization point. This happens repeatedly until the solver converges
// // to a consistent set of variable values. This requires us to specify an initial guess
// // for each variable, held in a Values container.
// #include <gtsam/nonlinear/Values.h>

// // Each variable in the system (poses and landmarks) must be identified with a
// // unique key. We can either use simple integer keys (1, 2, 3, ...) or symbols
// // (X1, X2, L1). Here we will use Symbols
// #include <gtsam/inference/Symbol.h>


// // utilities include functions for extracting poses from optimization results
// #include <gtsam/nonlinear/utilities.h>

// // // My custom utils
// // #include "include/msl_raptor_gtsam_utils.h"

// // Misc Includes
// #include <math.h>
// #include <algorithm>
// #include <random>
// #include <fstream>
// #include <iostream>


// We will use Pose3 variables (x, y, z, Rot3) to represent the robot/landmark positions
#include <gtsam/geometry/Pose3.h>

// In GTSAM, measurement functions are represented as 'factors'. Several common factors
// have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
// Here we will use Between factors for the relative motion described by odometry measurements.
// Also, we will initialize the robot at the origin using a Prior factor.
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/sam/BearingRangeFactor.h>

// When the factors are created, we will add them to a Factor Graph. As the factors we are using
// are nonlinear factors, we will need a Nonlinear Factor Graph.
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearEquality.h>  // used when we dont apriori know # of factors - https://gtsam.org/doxygen/index.html

// Finally, once all of the factors have been added to our factor graph, we will want to
// solve/optimize to graph to find the best (Maximum A Posteriori) set of variable values.
// GTSAM includes several nonlinear optimizers to perform this step. Here we will use the
// Levenberg-Marquardt solver
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

// Once the optimized values have been calculated, we can also calculate the marginal covariance
// of desired variables
#include <gtsam/nonlinear/Marginals.h>

// The nonlinear solvers within GTSAM are iterative solvers, meaning they linearize the
// nonlinear functions around an initial linearization point, then solve the linear system
// to update the linearization point. This happens repeatedly until the solver converges
// to a consistent set of variable values. This requires us to specify an initial guess
// for each variable, held in a Values container.
#include <gtsam/nonlinear/Values.h>

// Each variable in the system (poses and landmarks) must be identified with a
// unique key. We can either use simple integer keys (1, 2, 3, ...) or symbols
// (X1, X2, L1). Here we will use Symbols
#include <gtsam/inference/Symbol.h>

// We want to use iSAM2 incrementally, so include iSAM2 here
#include <gtsam/nonlinear/ISAM2.h>

// #include <Eigen/Core>
// #include <Eigen/Geometry>


// Misc Includes
#include <math.h>
#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>

using namespace std;
using namespace gtsam;

// Type Defs
typedef vector<tuple<double, Pose3>> object_data_vec_t;  // vector of tuples(double, pose message)
typedef tuple<double, int, Pose3, Pose3> data_tuple; 
typedef vector<data_tuple> object_est_gt_data_vec_t; //vector of tuples(double, int (id), pose message, pose message)

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

struct raptor_measurement_t {
  Symbol sym;
  Pose3 tf_w_ado_gt;
  Pose3 tf_w_ado_est;
  Pose3 tf_ego_ado_gt;
  Pose3 tf_ego_ado_est;
};

// Data Loading Helper Functions
void load_log_files(set<double> &times, object_est_gt_data_vec_t & ado_data, const string path, const string file_base, map<string, obj_param_t>, double dt_thresh);
void read_data_from_one_log(const string fn, object_data_vec_t& obj_data, set<double> & times);
void sync_est_and_gt(object_data_vec_t data_est, object_data_vec_t data_gt, object_est_gt_data_vec_t& ego_data, obj_param_t params, double dt_thresh);
void write_results_csv(string fn, 
                       map<Symbol, double> ego_time_map, 
                       map<Symbol, Pose3> tf_w_gt_map, 
                       map<Symbol, Pose3> tf_w_est_preslam_map, 
                       map<Symbol, Pose3> tf_w_est_postslam_map,
                       map<Symbol, map<Symbol, Pose3 > > tf_ego_ado_maps);
                      //  map<Symbol, map<Symbol, pair<Pose3, Pose3> > > tf_ego_ado_maps);

string pose_to_string_line(Pose3 p);

// Math Helper Functions
Pose3 add_init_est_noise(const Pose3 &ego_pose_est);
Rot3 remove_yaw(Rot3 R);
Pose3 remove_yaw(Pose3 P);
Eigen::Matrix3f rot3_to_matrix3f(Rot3 R);
Rot3 eigen_matrix3f_to_rot3(Eigen::Matrix3f M_in);
Eigen::Matrix3f create_rotation_matrix(float ax, float ay, float az);
void calc_pose_delta(const Pose3 & p1, const Pose3 &p2, double *trans_diff, double *rot_diff_rad, bool b_degrees);

