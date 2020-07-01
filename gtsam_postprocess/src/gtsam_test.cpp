/**
 * @file gtsam_test.cpp
 * @brief Attempt to read in data and process with isam
 * @author Adam Caccavale
 */

/**
 * Attempt to read in a data produced by MSL-RAPTOR and further process it with gtsam
 */

// We will use Pose3 variables (x, y, z, Rot3) to represent the robot/landmark positions
#include <gtsam/geometry/Pose3.h>

// In GTSAM, measurement functions are represented as 'factors'. Several common factors
// have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
// Here we will use Between factors for the relative motion described by odometry measurements.
// Also, we will initialize the robot at the origin using a Prior factor.
#include <gtsam/slam/BetweenFactor.h>

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

// utilities include functions for extracting poses from optimization results
#include <gtsam/nonlinear/utilities.h>

// Misc Includes
#include <math.h>
#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

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

// Loading & Processing Functions
void load_log_files(set<double> &times, object_est_gt_data_vec_t & ado_data, const string path, const string file_base, map<string, obj_param_t>, double dt_thresh);
void read_data_from_one_log(const string fn, object_data_vec_t& obj_data, set<double> & times);
void sync_est_and_gt(object_data_vec_t data_est, object_data_vec_t data_gt, object_est_gt_data_vec_t& ego_data, obj_param_t params, double dt_thresh);

// Helper functions
Pose3 add_init_est_noise(const Pose3 &ego_pose_est);
void calc_pose_delta(const Pose3 & p1, const Pose3 &p2, double *trans_diff, double *rot_diff_rad, bool b_degrees);
Rot3 remove_yaw(Rot3 R);
Pose3 remove_yaw(Pose3 P);
Eigen::Matrix3f rot3_to_matrix3f(Rot3 R);
Rot3 eigen_matrix3f_to_rot3(Eigen::Matrix3f M_in);
Eigen::Matrix3f create_rotation_matrix(float ax, float ay, float az);
void debug_func();
float fix_it(float f);

float fix_it(float f){
  if (f > 0){
    return M_PI - f;
  }
  else if (f < 0) {
    return -M_PI - f;
  }
  return f;
}

void debug_func() {
  Eigen::Vector3f ea;
  // float ax = -0.18336085, ay = -0.73772085, az = -0.00000104;// failure values: -0.18336085, -0.73772085, -0.00000104
  // float ax = -0.05, ay = -0.05, az = -0.05;
  float ax = -M_PI_4, ay = +M_PI_4, az = -M_PI_4;// failure values: -0.18336085, -0.73772085, -0.00000104
  Eigen::Matrix3f rot_matrix = create_rotation_matrix(ax, ay, az);
  cout << "\nax: " << ax << "(" << ax * 180.0/M_PI << "), ay: " << ay << "(" << ay*180.0/M_PI << "), az: " << az << "(" << az*180.0/M_PI << ") rad (in degrees)" << endl;
  cout << "rot_matrix:\n" << rot_matrix << endl;
  // int i = 0, j = 1, k = 2;
  int i = 0, j = 1, k = 2;
  ea = rot_matrix.eulerAngles(i, j, k); 
  cout << "order: " << i << j << k << " --> ea: " << ea[0] << " (" << ea[0]*180.0/M_PI << "), " << ea[1] << " (" << ea[1]*180.0/M_PI << "), " << ea[2] << " (" << ea[2]*180.0/M_PI << ")\n" << endl;
  
  
  cout << "    fixed --> ea: " << fix_it(ea[0]) << ", " << fix_it(ea[1]) << ", " << fix_it(ea[2]) << "\n" << endl;
  // i = 2; j = 1; k = 0;
  // ea = rot_matrix.eulerAngles(i, j, k); 
  // cout << "order: " << i << j << k << " --> ea: " << ea[0] << " (" << ea[0]*180.0/M_PI << "), " << ea[1] << " (" << ea[1]*180.0/M_PI << "), " << ea[2] << " (" << ea[2]*180.0/M_PI << ")\n" << endl;
  cout<<endl;
}

int main(int argc, char** argv) {
  // debug_func();
  // return 0;
  // useful gtsam examples:
  // https://github.com/borglab/gtsam/blob/develop/examples/VisualISAM2Example.cpp
  // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample.cpp
  // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample_large.cpp
  // Note: tf_A_B is a transform that when right multipled by a vector in frame B produces the same vector in frame A: p_A = tf_A_B * p_B

  double dt_thresh = 0.02; // how close a measurement is in time to ego pose to be "from" there - eventually should interpolate instead
  map<string, obj_param_t> obj_param_map = {
    {"bowl_white_small_norm", obj_param_t("bowl_white_small_norm", "bowl",   2, false, false, true)},
    {"camera_canon_len_norm", obj_param_t("camera_canon_len_norm", "camera", 3, false, false, false)},
    {"can_arizona_tea_norm",  obj_param_t("can_arizona_tea_norm",  "can",    4, false, false, true)},
    {"laptop_air_xin_norm",   obj_param_t("laptop_air_xin_norm",   "laptop", 5, false, false, false)},
    {"mug_daniel_norm",       obj_param_t("mug_daniel_norm",       "mug",    6, false, false, false)}
  };
  string path = "/mounted_folder/nocs_logs/";
  string base = "log_1_";

  object_est_gt_data_vec_t ado_data; // all ado data in 1 vector sorted by time (to be filled in by load_log_files)
  set<double> times;  // set of all unique times (to be filled in by load_log_files)
  load_log_files(times, ado_data, path, base, obj_param_map, dt_thresh);

  run_batch_slam(times, ado_data, obj_param_map, dt_thresh);
  // run_isam(times, ado_data, obj_param_map, dt_thresh);

  cout << "done with main!" << endl;
  return 0;
}

//////////////////////////////////////////////////////////
// Primary Slam Functions
//////////////////////////////////////////////////////////

void run_batch_slam(const set<double> &times, const object_est_gt_data_vec_t& obj_data, const map<std::string, obj_param_t> &obj_params_map, double dt_thresh) {
  // note: object id serves as landmark id, landmark is same as saying "ado"
  // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample.cpp <-- example VO, but using betweenFactors instead of stereo

  // STEP 0) Create graph & value objects, add first pose at origin with key '1'
  NonlinearFactorGraph graph;
  Pose3 first_pose = Pose3();
  int ego_pose_index = 1;
  Symbol ego_sym = Symbol('x', ego_pose_index);
  graph.emplace_shared<NonlinearEquality<Pose3> >(Symbol('x', ego_pose_index), first_pose);
  Values initial_estimate; // create Values object to contain initial estimates of camera poses and landmark locations

  // Eventually I will use the ukf's covarience here, but for now use a constant one
  auto constNoiseMatrix = noiseModel::Diagonal::Sigmas(
          (Vector(6) << Vector3::Constant(0.01), Vector3::Constant(0.03)).finished());

  int obj_list_ind = 0; // this needs to be flexible since I dont know how many landmarks I see at each time, if any
  bool b_landmarks_observed = false; // set to true if we observe at least 1 landmark (so we know if we should try to estimate a pose)
  
  // These will store the following poses: ground truth, quasi-odometry, and post-slam optimized
  map<Symbol, Pose3> tf_w_gt_map, tf_w_est_preslam_map, tf_w_est_postslam_map; // these are all tf_w_ego or tf_w_ado frames. Note: gt relative pose at t0 is the same as world pose (since we make our coordinate system based on our initial ego pose)
  
  // STEP 1) loop through ego poses, at each time do the following:
  //  - 1A) Add factors to graph between this pose and any visible landmarks various landmarks as we go 
  //  - 1B) If first timestep, use the fact that x1 is defined to be origin to set ground truth pose of landmarks. Also use first estimate as initial estimate for landmark pose
  //  - 1C) Otherwise, use gt / estimate of landmark poses from t0 and current time to gt / estimate of current camera pose (SUBSTITUE FOR ODOMETRY!!)
  //  - 1D) Use estimate of current camera pose for value initalization
  int t_ind = 0;
  for(const auto & ego_time : times) {
    Pose3 tf_w_ego_gt, tf_ego_ado_gt, tf_w_ego_est;
    ego_pose_index = 1 + t_ind;
    ego_sym = Symbol('x', ego_pose_index);

    if (obj_list_ind >= obj_data.size()) {
      cout << "encorperated all observations into graph" << endl;
      break;
    }
    
    // cout << "\n-----------------------------------------" << endl;
    // cout << "t_ind = " << t_ind << endl;
    vector<Rot3> r_vec;
    vector<Point3> t_vec;
    while(obj_list_ind < obj_data.size() && abs(get<0>(obj_data[obj_list_ind]) - ego_time) < dt_thresh ) {
      // DATA TYPE object_est_gt_data_vec_t: vector of tuples, each tuple is: <double time, int class id, Pose3 gt pose, Pose3 est pose>
      // first condition means we have more data to process, second means this observation is cooresponding with this ego pose
      b_landmarks_observed = true; // set to true because we have at least 1 object seen
      Pose3 tf_ego_ado_est = get<3>(obj_data[obj_list_ind]); // estimated ado pose
      int obj_id = get<1>(obj_data[obj_list_ind]);
      Symbol ado_sym = Symbol('l', obj_id);
      
      // 1A) - add ego pose <--> landmark (i.e. ado) pose factor. syntax is: ego_id ("x1"), ado_id("l3"), measurment (i.e. relative pose in ego frame tf_ego_ado_est), measurement uncertanty (covarience)
      graph.emplace_shared<BetweenFactor<Pose3> >(ego_sym, ado_sym, Pose3(tf_ego_ado_est), constNoiseMatrix);
      // cout << "creating factor " << ego_sym << " <--> " << ado_sym << endl;

      if(t_ind == 0) {
        // 1B) if first loop, assume all objects are seen and store their gt values - this will be used for intializing pose estimates
        tf_w_gt_map[ado_sym] = get<2>(obj_data[obj_list_ind]); // tf_w_ado_gt
        tf_w_est_preslam_map[ado_sym] = get<3>(obj_data[obj_list_ind]); // tf_w_ado_est

        // add initial estimate for landmark (in world frame)
        initial_estimate.insert(ado_sym, Pose3(tf_ego_ado_est)); // since by construction at t=0 the world and ego pose are both the origin, this relative measurement is also in world frame
        tf_w_ego_gt = Pose3();
      }
      else {
        // 1C) use gt position of landmark now & at t0 to get gt position of ego. Same for estimated position
        tf_ego_ado_gt = get<2>(obj_data[obj_list_ind]); // current relative gt object pose
        tf_w_ego_gt = tf_w_gt_map[ado_sym] * tf_ego_ado_gt.inverse(); // gt ego pose in world frame
        // if(false && obj_list_ind >= 199 && obj_list_ind< 204){
        //   cout << "at debug" << endl;
        //   Rot3 R = tf_ego_ado_gt.rotation();
        //   cout << R << endl;
        //   cout << R.xyz() << "\n"<< endl;
        //   Rot3 R2 = remove_yaw(R);
        //   cout << R2 << endl;
        //   cout << R2.xyz() << R2.yaw() << endl;
        //   cout << "end debug" << endl;
        // }
        tf_w_ego_est = tf_w_est_preslam_map[ado_sym] * tf_ego_ado_est.inverse(); // gt ego pose in world frame
        if (obj_id != 2 && obj_id!=4){ // assume we see at least 1 object w/o symmetry 
          r_vec.push_back(Rot3(tf_w_ego_est.rotation()));
        }
        t_vec.push_back(Point3(tf_w_ego_est.translation()));
      }
      obj_list_ind++;
    }
    if (b_landmarks_observed) {
      // 1D) only calculate our pose if we actually see objects
      initial_estimate.insert(ego_sym, Pose3(tf_w_ego_est));
      tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_est);
      tf_w_gt_map[ego_sym] = Pose3(tf_w_ego_gt);
    }
    b_landmarks_observed = false;
    t_ind++;
  }
  cout << "done building batch slam graph (w/ initializations)!" << endl;

  // STEP 2) create Levenberg-Marquardt optimizer for resulting factor graph, optimize
  LevenbergMarquardtOptimizer optimizer(graph, initial_estimate);
  Values result = optimizer.optimize();

  // STEP 3 Loop through each ego /landmark pose and compare estimate with ground truth (both before and after slam optimization)
  Values::ConstFiltered<Pose3> poses = result.filter<Pose3>();
  int i = 0;
  vector<double> t_diff_pre, rot_diff_pre, t_diff_post, rot_diff_post;
  double ave_t_diff_pre = 0, ave_rot_diff_pre = 0, ave_t_diff_post = 0, ave_rot_diff_post = 0;
  bool b_degrees = true;
  for(const auto& key_value: poses) {
    // Extract Symbol and Pose from dict & store in map
    Symbol sym = Symbol(key_value.key);
    if(sym == Symbol('x',44))
    {
      cout << "dsfs" << endl;
    }
    Pose3 tf_w_est_postslam = key_value.value;
    tf_w_est_postslam_map[sym] = tf_w_est_postslam;

    // Find corresponding gt pose and preslam pose
    Pose3 tf_w_gt = tf_w_gt_map[sym], tf_w_gt_inv = tf_w_gt.inverse();
    Pose3 tf_w_est_preslam = tf_w_est_preslam_map[sym];
    
    double t_diff_pre_val, rot_diff_pre_val, t_diff_post_val, rot_diff_post_val; 
    calc_pose_delta(tf_w_est_preslam, tf_w_gt_inv, &t_diff_pre_val, &rot_diff_pre_val, b_degrees);
    calc_pose_delta(tf_w_est_postslam, tf_w_gt_inv, &t_diff_post_val, &rot_diff_post_val, b_degrees);
    t_diff_pre.push_back(t_diff_pre_val);
    rot_diff_pre.push_back(rot_diff_pre_val);
    t_diff_post.push_back(t_diff_post_val);
    rot_diff_post.push_back(rot_diff_post_val);
    ave_t_diff_pre += abs(t_diff_pre_val);
    ave_rot_diff_pre += abs(rot_diff_pre_val);
    ave_t_diff_post += abs(t_diff_post_val);
    ave_rot_diff_post += abs(rot_diff_post_val);

    cout << "-----------------------------------------------------" << endl;
    cout << "evaluating " << sym << ":" << endl;
    // cout << "gt pose:" << tf_w_gt << endl;
    // cout << "pre-process est pose:" << tf_w_est_preslam << endl;
    // cout << "post-process est pose:" << tf_w_est_postslam << endl;
    cout << "delta pre-slam: t = " << t_diff_pre_val << ", ang = " << rot_diff_pre_val << " deg" << endl;
    cout << "delta post-slam: t = " << t_diff_post_val << ", ang = " << rot_diff_post_val << " deg" << endl;

    i++;
  }
  ave_t_diff_pre /= double(i);
  ave_rot_diff_pre /= double(i);
  ave_t_diff_post /= double(i);
  ave_rot_diff_post /= double(i);
  cout << "\n-----------------------------------------------------\n-----------------------------------------------------\n" << endl;
  cout << "initial error = " << graph.error(initial_estimate) << endl;  // iterate over all the factors_ to accumulate the log probabilities
  cout << "final error = " << graph.error(result) << "\n" << endl;  // iterate over all the factors_ to accumulate the log probabilities
  cout << "Averages t_pre = " << ave_t_diff_pre << ", t_post = " << ave_t_diff_post << endl;
  cout << "Averages rot_pre = " << ave_rot_diff_pre << " deg, rot_post = " << ave_rot_diff_post << " deg" << endl;
}

void run_isam(const set<double> &times, const object_est_gt_data_vec_t& obj_data, const map<std::string, obj_param_t> &obj_params_map, double dt_thresh) {
  // // parameters
  // size_t minK = floor(times.size()*0.1); // minimum number of range measurements to process initially
  // size_t incK = 1; // how often do we update the slam map? 1 - every time, 2 - every other etc etc

  // NonlinearFactorGraph newFactors;
  // Values initial_estimate; // create Values object to contain initial estimates of camera poses and landmark locations

  // // Set Noise parameters
  // Vector priorSigmas = Vector3(1,1,M_PI);
  // Vector odoSigmas = Vector3(0.05, 0.01, 0.1);
  // double sigmaR = 100; // range standard deviation
  // // const NM::Base::shared_ptr // all same type
  // // priorNoise = NM::Diagonal::Sigmas(priorSigmas), //prior
  // // odoNoise = NM::Diagonal::Sigmas(odoSigmas), // odometry
  
  // // Eventually I will use the ukf's covarience here, but for now use a constant one
  // auto raptorNoise = noiseModel::Diagonal::Sigmas(
  //         (Vector(6) << Vector3::Constant(0.01), Vector3::Constant(0.03)).finished());

  // // Initialize iSAM
  // ISAM2 isam;

  // // Create first pose exactly at origin // NOTE: might want to do this in the future: Add prior on first pose // newFactors.addPrior(0, pose0, priorNoise);
  // Pose3 first_pose = Pose3();
  // int ego_pose_index = 1;
  // Symbol ego_sym = Symbol('x', ego_pose_index);
  // newFactors.emplace_shared<NonlinearEquality<Pose3> >(Symbol('x', ego_pose_index), first_pose);
  // initial_estimate.insert(ego_sym, first_pose);

  // //  initialize points
  // initial_estimate.insert(symbol('L', 1), Point2(3.5784, 2.76944));
  // initial_estimate.insert(symbol('L', 6), Point2(-1.34989, 3.03492));
  // initial_estimate.insert(symbol('L', 0), Point2(0.725404, -0.0630549));
  // initial_estimate.insert(symbol('L', 5), Point2(0.714743, -0.204966));


  // // set some loop variables
  // size_t i = 1; // step counter
  // size_t k = 0; // relative pose measurement counter
  // bool initialized = false;
  // Pose3 lastPose = first_pose;
  // size_t countK = 0;
  // int obj_list_ind = 0; // this needs to be flexible since I dont know how many landmarks I see at each time, if any

  // // These will store the following poses: ground truth, quasi-odometry, and post-slam optimized
  // // map<Symbol, Pose3> tf_w_gt_map, tf_w_est_preslam_map, tf_w_est_postslam_map; // these are all tf_w_ego or tf_w_ado frames. Note: gt relative pose at t0 is the same as world pose (since we make our coordinate system based on our initial ego pose)
  // map<Symbol, Pose3> tf_w_gt_t0_map, tf_w_est_t0_map;
  // int t_ind = 0;
  // for(const auto & ego_time : times) { 
  //   Pose3 tf_w_ego_gt, tf_ego_ado_gt, tf_w_ego_est;
  //   ego_pose_index = 1 + t_ind;
  //   ego_sym = Symbol('x', ego_pose_index);

  //   if (obj_list_ind >= obj_data.size()) {
  //     cout << "DONE WITH DATA - encorperated all observations into graph" << endl;
  //     break;
  //   }
  //   vector<raptor_measurement_t> raptor_measurement_vec;
  //   simulate_measurement_from_bag(ego_time, t_ind, obj_list_ind, obj_data, obj_params_map, dt_thresh, tf_w_gt_t0_map, tf_w_est_t0_map, tf_w_ego_gt, tf_w_ego_est, raptor_measurement_vec);
  //   if(tf_w_gt_t0_map.find(ego_sym) == tf_w_gt_t0_map.end()){ // key was not in map --> add it
  //     tf_w_gt_t0_map[ego_sym] = tf_w_ego_gt;
  //     tf_w_est_t0_map[ego_sym] = tf_w_ego_est;
  //   }
  //   // each of these is a a "measurement" of a landmark. Add a factor to the graph & an estimate of its value
  //   for(const auto & meas : raptor_measurement_vec) {
  //     // key was not in map --> add it (this is first time this object was seen)
  //     if(tf_w_gt_t0_map.find(meas.sym) == tf_w_gt_t0_map.end()){ 
  //       tf_w_gt_t0_map[meas.sym] = meas.tf_w_ado_gt;
  //       tf_w_est_t0_map[meas.sym] = meas.tf_w_ado_est;
  //     }

  //     // add factor to graph
  //     newFactors.emplace_shared<BetweenFactor<Pose3> >(ego_sym, meas.sym, meas.tf_ego_ado_est, raptorNoise);
  //   }
  // }
/*
  // Loop over odometry
  for(const TimedOdometry& timedOdometry: odometry) {
    //--------------------------------- odometry loop -----------------------------------------
    double t;
    Pose2 odometry;
    boost::tie(t, odometry) = timedOdometry;

    // add odometry factor
    newFactors.push_back(BetweenFactor<Pose2>(i-1, i, odometry, odoNoise));

    // predict pose and add as initial estimate
    Pose2 predictedPose = lastPose.compose(odometry);
    lastPose = predictedPose;
    initial_estimate.insert(i, predictedPose);

    // Check if there are range factors to be added
    while (k < K && t >= boost::get<0>(triples[k])) {
      size_t j = boost::get<1>(triples[k]);
      double range = boost::get<2>(triples[k]);
      newFactors.push_back(RangeFactor<Pose2, Point2>(i, symbol('L', j), range,rangeNoise));
      k = k + 1;
      countK = countK + 1;
    }

    // Check whether to update iSAM 2
    if ((k > minK) && (countK > incK)) {
      if (!initialized) { // Do a full optimize for first minK ranges
        LevenbergMarquardtOptimizer batchOptimizer(newFactors, initial_estimate);
        initial_estimate = batchOptimizer.optimize();
        initialized = true;
      }
      isam.update(newFactors, initial_estimate);
      Values result = isam.calculateEstimate();
      lastPose = result.at<Pose2>(i);
      newFactors = NonlinearFactorGraph();
      initial_estimate = Values();
      countK = 0;
    }
    i += 1;
    //--------------------------------- odometry loop -----------------------------------------
  } // end for
*/
}

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
                                  vector<raptor_measurement_t> &raptor_measurement_vec) { // output
  // double ego_time: current simulation time in seconds
  // int t_ind: an int tracking which index of the set of times we are on
  // int obj_list_ind: an int tracking where we are in our list of data (sorted by time, then by object id)
  // object_est_gt_data_vec_t obj_data: the full map containing all observations (the keys are the times)
  // map<std::string, obj_param_t> obj_params_map: parameter for each object
  // double dt_thresh: how close in time a measurement has to be to the pose estimate to be considered the same (eventually interpolate?)
  
  int ego_pose_index = 1 + t_ind;
  Symbol ego_sym = Symbol('x', ego_pose_index);
  while(obj_list_ind < obj_data.size() && abs(get<0>(obj_data[obj_list_ind]) - ego_time) < dt_thresh ) {
      // DATA TYPE object_est_gt_data_vec_t: vector of tuples, each tuple is: <double time, int class id, Pose3 gt pose, Pose3 est pose>
      // first condition means we have more data to process, second means this observation is cooresponding with this ego pose
      raptor_measurement_t meas;
      meas.sym = Symbol('l', get<1>(obj_data[obj_list_ind]));
      // tuple<Symbol, Pose3, Pose3> tf_w_ado_data, tf_ego_ado_data;

      meas.tf_ego_ado_gt = get<2>(obj_data[obj_list_ind]);
      meas.tf_ego_ado_est = get<3>(obj_data[obj_list_ind]); 
      
      // dont fill in map here, that will be done outside this function
      if(t_ind == 0) {
        tf_w_ego_gt = Pose3();
        tf_w_ego_est = Pose3();  // since we define our initial pose as the origin, this is always "right"
        meas.tf_w_ado_gt = Pose3(meas.tf_ego_ado_gt);
        meas.tf_w_ado_est = Pose3(meas.tf_ego_ado_est);
        raptor_measurement_vec.push_back(meas);
      }
      else {
        if(tf_w_gt_t0_map.find(meas.sym) == tf_w_gt_t0_map.end()){ // this is the first time we are seeing this object
          // in this situation we need to establish tf_w_ego from the other landmarks. if there are no other landmarks we should either save this for later or skip it
          // tf_w_ego <-- from other landmarks
          // tf_w_ado = tf_w_ego * tf_ego_ado
          runtime_error("We do not handle the case where we dont see all objects on first timestep yet");
        }
        tf_w_ego_gt = tf_w_gt_t0_map[meas.sym] * (meas.tf_ego_ado_gt).inverse(); // gt ego pose in world frame
        tf_w_ego_est = tf_w_est_t0_map[meas.sym] * (meas.tf_ego_ado_est).inverse(); // gt ego pose in world frame
        meas.tf_w_ado_gt = tf_w_ego_gt * meas.tf_ego_ado_gt; 
        meas.tf_w_ado_est = tf_w_ego_est * meas.tf_ego_ado_est; 

        // tf_w_ado_vec.push_back( make_tuple(meas.sym, tf_w_ado_gt, tf_w_ado_est) );
        // tf_ego_ado_vec.push_back( make_tuple(meas.sym, get<2>(obj_data[obj_list_ind]), get<3>(obj_data[obj_list_ind])) );
        raptor_measurement_vec.push_back(meas);

      }
      obj_list_ind++;
    }
}

//////////////////////////////////////////////////////////
// Data Loading Helper Functions
//////////////////////////////////////////////////////////

void load_log_files(set<double> &times, object_est_gt_data_vec_t & ado_data, const string path, const string file_base, map<string, obj_param_t> obj_params, double dt_thresh) {
  // for each object, load its est and gt log files to extract pose and time information. combine into a set of all times, and also all the data sorted by time
  vector<object_data_vec_t> ado_data_gt, ado_data_est;
  for(const auto &key_value_pair : obj_params) {
    object_data_vec_t ado_data_gt, ado_data_est;
    object_est_gt_data_vec_t ado_data_single;
    obj_param_t params = key_value_pair.second;
    // string obj_long_name = params.long_name; //key_value_pair.first;
    // int obj_id = params.obj_id; //object_id_map[key_value_pair.second];
    cout << "Processing " << params.long_name << " (id = " << params.obj_id << ")" << endl;

    read_data_from_one_log(path + file_base + params.long_name + "_gt.log", ado_data_gt, times);
    read_data_from_one_log(path + file_base + params.long_name + "_est.log", ado_data_est, times);
    sync_est_and_gt(ado_data_est, ado_data_gt, ado_data_single, params, dt_thresh);
    ado_data.insert( ado_data.end(), ado_data_single.begin(), ado_data_single.end() ); // combine into 1 vector of all ado data
  }

  // sort by time & object id (the later is just for readability of debugging output, is only tie-break if time are equal)
  std::sort(ado_data.begin(), ado_data.end(), [](const data_tuple& lhs, const data_tuple& rhs) {
    if (get<0>(lhs) == get<0>(rhs)) { // if times are equal
      return get<1>(lhs) < get<1>(rhs); // return true if lhs has lower id
    }
    return get<0>(lhs) < get<0>(rhs); 
  });   // this should sort the vector by time (i.e. first element in each tuple). Tie breaker is object id
}

void read_data_from_one_log(const string fn, object_data_vec_t& obj_data, set<double> &times){
  // log file header: Time (s), Ado State GT, Ego State GT, 3D Corner GT (X|Y|Z), Corner 2D Projections GT (r|c), Angled BB (r|c|w|h|ang_deg), Image Segmentation Mode
  // note: the states are position (3), lin vel (3), quat wxyz (4), ang vel (3) (space deliminated)
  ifstream infile(fn);
  string line, s;
  double time, x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz;
  Pose3 pose;
  getline(infile, line); // skip header of file
  while (getline(infile, line)) {
    istringstream iss(line);
    iss >> time;
    iss >> x;
    iss >> y;
    iss >> z;
    iss >> vx;
    iss >> vy;
    iss >> vz;
    iss >> qw;
    iss >> qx;
    iss >> qy;
    iss >> qz;
    iss >> wx;
    iss >> wy;
    iss >> wz;
    pose = Pose3(Rot3(Quaternion(qw, qx, qy, qz)), Point3(x, y, z));
    times.insert(time);
    obj_data.push_back(make_tuple(time, pose));
    // obj_data.push_back(make_tuple(time, remove_yaw(pose)));
    continue;
  }
}

void sync_est_and_gt(object_data_vec_t data_est, object_data_vec_t data_gt, object_est_gt_data_vec_t& data, obj_param_t params, double dt_thresh) {
  // data.push_back(make_tuple(5, get<1>(data_est[0]), get<1>(data_est[0])));
  // now "sync" the gt and est for each object
  
  double t_gt, t_est;
  uint next_est_time_ind = 0;
  for (uint i = 0; i < data_gt.size(); i++) {
    t_gt = get<0>(data_gt[i]);
    for (uint j = next_est_time_ind; j < data_est.size(); j++) {
      t_est = get<0>(data_est[j]);
      if(dt_thresh > abs(t_gt - t_est)) {
        data.push_back(make_tuple((t_gt + t_est)/2, params.obj_id, get<1>(data_gt[i]), get<1>(data_est[j])));
        if (params.obj_id != 1) { // no pose data for ego robot, all zeros
          double t_diff, rot_diff; 
          calc_pose_delta(get<1>(data_gt[i]).inverse(), get<1>(data_est[j]), &t_diff, &rot_diff, true);

          cout << "\n-------------------------------------------------------------" << endl;
          cout << "a) time = " << t_est << ". id = " << params.obj_id << ".  gt / est diff:  t_delta = " << t_diff << ", r_delta = " << rot_diff << " deg" << endl;
          
          double t_diff2, rot_diff2;
          if (!params.b_rm_roll && !params.b_rm_pitch && !params.b_rm_yaw) {
            calc_pose_delta(get<1>(data_gt[i]).inverse(), get<1>(data_est[j]), &t_diff2, &rot_diff2, true);
            cout << "b) \t\t\t   not symetric" << endl;
          }
          else {
            if (params.b_rm_roll) {
              runtime_error("Need to implement remove_roll()!");
            }
            if (params.b_rm_pitch) {
              runtime_error("Need to implement remove_pitch()!");
            }
            if (params.b_rm_yaw) {
              Pose3 data_gt_no_yaw = remove_yaw(get<1>(data_gt[i]));
              Pose3 data_est_no_yaw = remove_yaw(get<1>(data_est[i]));
              calc_pose_delta(data_gt_no_yaw.inverse(), data_est_no_yaw, &t_diff2, &rot_diff2, true);
              cout << "c) \t\t\t   w/o yaw:  t_delta2 = " << t_diff2 << ", r_delta2 = " << rot_diff2 << " deg" << endl;
              calc_pose_delta(data_gt_no_yaw.inverse(), get<1>(data_gt[j]), &t_diff2, &rot_diff2, true);
              cout << "d) \t\t\t   w/ vs. w/o yaw [gt]:   t_diff = " << t_diff2 << ", r_diff = " << rot_diff2 << " deg" << endl;
              calc_pose_delta(data_est_no_yaw.inverse(), get<1>(data_est[j]), &t_diff2, &rot_diff2, true);
              cout << "e) \t\t\t   w/ vs. w/o yaw [est]:  t_diff = " << t_diff2 << ", r_diff = " << rot_diff2 << " deg" << endl;

              if (t_est > 31.7 && params.obj_id == 2 && t_est < 31.9) {
                cout << "f) gt yaw: "<< get<1>(data_gt[i]) << endl;              
                cout << "g) gt no yaw: " << data_gt_no_yaw << endl;
                cout << endl;  
              }
            }
          }
          
        }
        next_est_time_ind = j + 1;
        break;
      }
    }
  }
  cout << endl;
}


//////////////////////////////////////////////////////////
// Math Helper Functions
//////////////////////////////////////////////////////////

Pose3 add_init_est_noise(const Pose3 &ego_pose_est) {
  // noise = np.array([random.uniform(-0.02, 0.02) for i in range(3)]) 
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-0.00005, 0.00005);
  // Pose3 delta(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(dis(gen), dis(gen), dis(gen)));
  Pose3 delta(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(0.00,0.00,0.00));
  cout << "noise:" << delta << "tf before noise: " << ego_pose_est << endl;
  return ego_pose_est * delta;
  // ego_pose_est = ego_pose_est.compose(delta);
  
  // cout << "tf after noise: " << ego_pose_est << endl;

}

Pose3 remove_yaw(Pose3 P) {
  return Pose3(remove_yaw(P.rotation()), P.translation());
}

Rot3 remove_yaw(Rot3 R) {
  // Goal - recover the XYZ euler angles and set the yaw (rotation about Z) to 0
  // https://eigen.tuxfamily.org/dox/group__Geometry__Module.html#ga17994d2e81b723295f5bc3b1f862ed3b  || https://stackoverflow.com/questions/31589901/euler-to-quaternion-quaternion-to-euler-using-eigen
  /// https://stackoverflow.com/questions/31589901/euler-to-quaternion-quaternion-to-euler-using-eigen
  // https://eigen.tuxfamily.org/dox/group__Geometry__Module.html
  // https://eigen.tuxfamily.org/dox/classEigen_1_1AngleAxis.html
  // roll (X) pitch (Y) yaw (Z) (set Z to 0)


  // Eigen::Vector3f ea = rot3_to_matrix3f(R).eulerAngles(1, 1, 2); 
  // cout << "1) R as ea: " << ea[0] << ", " << ea[1] << ", " << ea[2] << "\n" << endl;
  // Eigen::Quaternionf Q_no_yaw = Eigen::AngleAxisf(ea[0], Eigen::Vector3f::UnitX()) * 
  //                               Eigen::AngleAxisf(ea[1], Eigen::Vector3f::UnitY()) * 
  //                               Eigen::AngleAxisf(0.0,   Eigen::Vector3f::UnitZ());
  // Rot3 R_out = Rot3(Quaternion(Q_no_yaw.w(), Q_no_yaw.x(), Q_no_yaw.y(), Q_no_yaw.z()));
  // Eigen::Vector3f ea_out = rot3_to_matrix3f(R_out).eulerAngles(1, 1, 2); 
  // cout << "2) R_out as ea: " << ea_out[0] << ", " << ea_out[1] << ", " << ea_out[2] << "\n" << endl;
  // return R_out;


  // https://www.mecademic.com/resources/Euler-angles/Euler-angles
  float alpha, beta, gamma;
  Matrix3 M = R.matrix(), M_out;
  double x,y,z,cx,cy,cz,sx,sy,sz;
  sy = M(0,2);

  if ( abs(abs(M(0,2)) - 1) > 0.0001) {
    // not at a singularity
    beta = asin(M(0,2));
    gamma = atan2(-M(0,1), M(0,0));
    alpha = atan2(-M(1,2), M(2,2));
  }
  else {
    alpha = 0;
    beta = M_PI_2;
    gamma = atan2(M(1,0), M(1,1));
  }
  Eigen::Matrix3f rot_matrix = create_rotation_matrix(alpha, beta, 0.0);
  return eigen_matrix3f_to_rot3(rot_matrix);
}

void calc_pose_delta(const Pose3 & p1, const Pose3 &p2, double *trans_diff, double *rot_diff_rad, bool b_degrees){
  // b_degrees is true if we want degrees, false for radians 
  Pose3 delta = p1.compose(p2);
  *trans_diff = delta.translation().squaredNorm();
  double thresh = 0.001;
  double unit_multiplier = 1;
  double acos_input = (delta.rotation().matrix().trace() - 1) / 2.0;
  if (b_degrees){
    unit_multiplier = 180.0 / M_PI;
  }
  if (acos_input > 1 && (acos_input - 1) < thresh) {
    *rot_diff_rad = 0;
  }
  else if (acos_input > 1) {
    runtime_error("ERROR: cant have acos input > 1!!");
  }
  else if (acos_input < -1 && (abs(acos_input) - 1) < thresh){
    *rot_diff_rad = M_PI * unit_multiplier;
  }
  else if (acos_input < -1 && (abs(acos_input) - 1) < thresh){
    runtime_error("ERROR: cant have acos input < -1!!");
  }
  else {
    *rot_diff_rad = acos( acos_input ) * unit_multiplier;
  }
}

Eigen::Matrix3f rot3_to_matrix3f(Rot3 R) {
  Eigen::Matrix3f m;
  m(0,0) = R.matrix()(0,0); m(0,1) = R.matrix()(0,1); m(0,2) = R.matrix()(0,2);
  m(1,0) = R.matrix()(1,0); m(1,1) = R.matrix()(1,1); m(1,2) = R.matrix()(1,2);
  m(2,0) = R.matrix()(2,0); m(2,1) = R.matrix()(2,1); m(2,2) = R.matrix()(2,2);
  return m;
}

Eigen::Matrix3f create_rotation_matrix(float ax, float ay, float az) {

  // Eigen::Matrix3f R_deltax = np.array([[ 1.             , 0.             , 0.              ],
  //                         [ 0.             , np.cos(Angle_x),-np.sin(Angle_x) ],
  //                         [ 0.             , np.sin(Angle_x), np.cos(Angle_x) ]]);
  // Eigen::Matrix3f R_deltay = np.array([[ np.cos(Angle_y), 0.             , np.sin(Angle_y) ],
  //                         [ 0.             , 1.             , 0               ],
  //                         [-np.sin(Angle_y), 0.             , np.cos(Angle_y) ]]);
  // Eigen::Matrix3f R_deltaz = np.array([[ np.cos(Angle_z),-np.sin(Angle_z), 0.              ],
  //                         [ np.sin(Angle_z), np.cos(Angle_z), 0.              ],
  //                         [ 0.             , 0.             , 1.              ]]);

  Eigen::Matrix3f R_deltax, R_deltay, R_deltaz;
  R_deltax << 1, 0, 0, 0, cos(ax), -sin(ax), 0, sin(ax), cos(ax);
  R_deltay << cos(ay), 0, sin(ay), 0, 1, 0, -sin(ay), 0, cos(ay);
  R_deltaz << cos(az), -sin(az), 0, sin(az), cos(az), 0, 0, 0, 1;
  return R_deltax * R_deltay * R_deltaz;
}


Rot3 eigen_matrix3f_to_rot3(Eigen::Matrix3f M_in) {
  Matrix3 M_out;
  M_out << M_in(0,0), M_in(0,1), M_in(0,2), M_in(1,0), M_in(1,1), M_in(1,2), M_in(2,0), M_in(2,1), M_in(2,2);
  return Rot3(M_out);
}


// bool areQuaternionsClose(Quaternion q1, Quaternion q2){
// 	float dot = q1.dot(22); // cos(theta / 2)
// 	if(dot < 0.0f){
// 		return false;					
// 	}
// 	else{
// 		return true;
// 	}
// }

// void average_poses(vector<Pose3>, p_vec){



// }

// void average_quats(Quaternion &cumulative, Quaternion newRotation, Quaternion firstRotation, int addAmount) {
//   //http://wiki.unity3d.com/index.php/Averaging_Quaternions_and_Vectors
//   //Get an average (mean) from more then two quaternions (with two, slerp would be used).
//   //Note: this only works if all the quaternions are relatively close together.
//   //Usage: 
//   //-Cumulative is an ___ which holds all the added x y z and w components.
//   //-newRotation is the next rotation to be added to the average pool
//   //-firstRotation is the first quaternion of the array to be averaged
//   //-addAmount holds the total amount of quaternions which are currently added
//   //This function returns the current average quaternion

// 	float w = 0.0f;
// 	float x = 0.0f;
// 	float y = 0.0f;
// 	float z = 0.0f;
 
// 	//Before we add the new rotation to the average (mean), we have to check whether the quaternion has to be inverted. Because
// 	//q and -q are the same rotation, but cannot be averaged, we have to make sure they are all the same.
// 	if(!areQuaternionsClose(newRotation, firstRotation)){
// 		newRotation = Quaternion(-newRotation.w, -newRotation.x, -newRotation.y, -newRotation.z);	
// 	}
 
// 	//Average the values
// 	float addDet = 1f/(float)addAmount;
// 	cumulative.w += newRotation.w;
// 	w = cumulative.w * addDet;
// 	cumulative.x += newRotation.x;
// 	x = cumulative.x * addDet;
// 	cumulative.y += newRotation.y;
// 	y = cumulative.y * addDet;
// 	cumulative.z += newRotation.z;
// 	z = cumulative.z * addDet;		
// }