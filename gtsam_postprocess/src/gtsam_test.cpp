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

using namespace std;
using namespace gtsam;

// Type Defs for loading data
typedef vector<tuple<double, Pose3>> object_data_vec_t;  // vector of tuples(double, pose message)
typedef tuple<double, int, Pose3, Pose3> data_tuple; 
typedef vector<data_tuple> object_est_gt_data_vec_t; //vector of tuples(double, int (id), pose message, pose message)

// Loading & Processing Functions
void load_log_files(set<double> &times, object_est_gt_data_vec_t & ado_data, const string path, const string file_base, map<string, string> & object_long_to_short_name, map<string, int> & object_id_map, double dt_thresh);
void read_data_from_one_log(const string fn, object_data_vec_t& obj_data, set<double> & times);
void sync_est_and_gt(object_data_vec_t data_est, object_data_vec_t data_gt, object_est_gt_data_vec_t& ego_data, int object_id, double dt_thresh);

// GTSAM-RAPTOR Function
void run_isam(object_est_gt_data_vec_t& all_data, map<std::string, int> &object_id_map);
void run_batch_slam(const set<double> &times, const object_est_gt_data_vec_t& obj_data, const map<std::string, int> &object_id_map, double dt_thresh);

// Helper functions
Pose3 add_init_est_noise(const Pose3 &ego_pose_est);
void calc_pose_delta(const Pose3 & p1, const Pose3 &p2, double *trans_diff, double *rot_diff_rad, bool b_degrees);
Rot3 remove_yaw(Rot3 R);
Pose3 remove_yaw(Pose3 P);

int main(int argc, char** argv) {
  // useful gtsam examples:
  // https://github.com/borglab/gtsam/blob/develop/examples/VisualISAM2Example.cpp
  // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample.cpp
  // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample_large.cpp
  // Note: tf_A_B is a transform that when right multipled by a vector in frame B produces the same vector in frame A: p_A = tf_A_B * p_B

  double dt_thresh = 0.02; // how close a measurement is in time to ego pose to be "from" there - eventually should interpolate instead
  map<std::string, int> object_id_map = {
        {"ego", 1},
        {"bowl", 2},
        {"camera", 3},
        {"can", 4},
        {"laptop", 5},
        {"mug", 6}
  };
  map<string, string> object_long_to_short_name = {
    {"bowl_white_small_norm", "bowl"},
    { "camera_canon_len_norm", "camera"},
    {"can_arizona_tea_norm", "can"},
    {"laptop_air_xin_norm", "laptop"},
    {"mug_daniel_norm", "mug"}
  };
  string path = "/mounted_folder/nocs_logs/";
  string base = "log_1_";

  object_est_gt_data_vec_t ado_data; // all ado data in 1 vector sorted by time (to be filled in by load_log_files)
  set<double> times;  // set of all unique times (to be filled in by load_log_files)
  load_log_files(times, ado_data, path, base, object_long_to_short_name, object_id_map, dt_thresh);

  run_batch_slam(times, ado_data, object_id_map, dt_thresh);
  // run_isam(all_data, object_id_map);

  cout << "done with main!" << endl;
  return 0;
}

void load_log_files(set<double> &times, object_est_gt_data_vec_t & ado_data, const string path, const string file_base, map<string, string> & object_long_to_short_name, map<string, int> & object_id_map, double dt_thresh) {
  // for each object, load its est and gt log files to extract pose and time information. combine into a set of all times, and also all the data sorted by time
  vector<object_data_vec_t> ado_data_gt, ado_data_est;
  for(const auto &key_value_pair : object_long_to_short_name) {
    object_data_vec_t ado_data_gt, ado_data_est;
    object_est_gt_data_vec_t ado_data_single;
    string obj_long_name = key_value_pair.first;
    int obj_id = object_id_map[key_value_pair.second];

    read_data_from_one_log(path + file_base + obj_long_name + "_gt.log", ado_data_gt, times);
    read_data_from_one_log(path + file_base + obj_long_name + "_est.log", ado_data_est, times);
    sync_est_and_gt(ado_data_est, ado_data_gt, ado_data_single, obj_id, dt_thresh);
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
    obj_data.push_back(make_tuple(time, remove_yaw(pose)));
    continue;
  }
}

void run_batch_slam(const set<double> &times, const object_est_gt_data_vec_t& obj_data, const map<std::string, int> &object_id_map, double dt_thresh) {
  // note: object id serves as landmark id, landmark is same as saying "ado"
  // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample.cpp <-- example VO, but using betweenFactors instead of stereo

  // STEP 0) Create graph & value objects, add first pose at origin with key '1'
  NonlinearFactorGraph graph;
  Pose3 first_pose;
  int ego_pose_index = 1;
  Symbol ego_sym = Symbol('x', ego_pose_index);
  graph.emplace_shared<NonlinearEquality<Pose3> >(Symbol('x', ego_pose_index), Pose3());
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
    if(ego_pose_index == 44){
      cout << "trouble case" << endl;
    }

    if (obj_list_ind >= obj_data.size()) {
      cout << "encorperated all observations into graph" << endl;
      break;
    }
    
    // cout << "\n-----------------------------------------" << endl;
    // cout << "t_ind = " << t_ind << endl;
    while(obj_list_ind < obj_data.size() && abs(get<0>(obj_data[obj_list_ind]) - ego_time) < dt_thresh ) {
      // DATA TYPE object_est_gt_data_vec_t: vector of tuples, each tuple is: <double time, int class id, Pose3 gt pose, Pose3 est pose>
      // first condition means we have more data to process, second means this observation is cooresponding with this ego pose
      b_landmarks_observed = true; // set to true because we have at least 1 object seen
      Pose3 tf_ego_ado_est = get<3>(obj_data[obj_list_ind]); // estimated ado pose
      Symbol ado_sym = Symbol('l', get<1>(obj_data[obj_list_ind]));
      
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
        if(obj_list_ind >= 199 && obj_list_ind< 204){
          cout << "at debug" << endl;
          Rot3 R = tf_ego_ado_gt.rotation();
          cout << R << endl;
          cout << R.xyz() << "\n"<< endl;
          Rot3 R2 = remove_yaw(R);
          cout << R2 << endl;
          cout << R2.xyz() << R2.yaw() << endl;
          cout << "end debug" << endl;
        }
        tf_w_ego_est = tf_w_est_preslam_map[ado_sym] * tf_ego_ado_est.inverse(); // gt ego pose in world frame
      }
      obj_list_ind++;
    }
    if (b_landmarks_observed)
      // 1D) only calculate our pose if we actually see objects
      initial_estimate.insert(ego_sym, Pose3(tf_w_ego_est));
      tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_est);
      tf_w_gt_map[ego_sym] = Pose3(tf_w_ego_gt);
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
    ave_t_diff_pre += t_diff_pre_val;
    ave_rot_diff_pre += rot_diff_pre_val;
    ave_t_diff_post += t_diff_post_val;
    ave_rot_diff_post += rot_diff_post_val;

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

void calc_pose_delta(const Pose3 & p1, const Pose3 &p2, double *trans_diff, double *rot_diff_rad, bool b_degrees){
  // b_degrees is true if we want degrees, false for radians 
  Pose3 delta = p1.compose(p2);
  *trans_diff = delta.translation().squaredNorm();
  double tmp = (delta.rotation().matrix().trace() - 1) / 2.0;
  double thresh = 0.001;
  double unit_multiplier = 1;
  if (b_degrees){
    unit_multiplier = 180.0 / M_PI;
  }
  if (tmp > 1 && (tmp - 1) < thresh) {
    *rot_diff_rad = 0;
  }
  else if (tmp > 1) {
    runtime_error("ERROR: cant have acos input > 1!!");
  }
  else if (tmp < -1 && (abs(tmp) - 1) < thresh){
    *rot_diff_rad = M_PI * unit_multiplier;
  }
  else if (tmp < -1 && (abs(tmp) - 1) < thresh){
    runtime_error("ERROR: cant have acos input < -1!!");
  }
  else {
    *rot_diff_rad = acos( tmp ) * unit_multiplier;
  }
}

void run_isam(object_est_gt_data_vec_t& all_data, map<std::string, int> &object_id_map) {
  // // Create an iSAM2 object. Unlike iSAM1, which performs periodic batch steps
  // // to maintain proper linearization and efficient variable ordering, iSAM2
  // // performs partial relinearization/reordering at each step. A parameter
  // // structure is available that allows the user to set various properties, such
  // // as the relinearization threshold and type of linear solver. For this
  // // example, we we set the relinearization threshold small so the iSAM2 result
  // // will approach the batch result.
  // ISAM2Params parameters;
  // parameters.relinearizeThreshold = 0.01;
  // parameters.relinearizeSkip = 1;
  // ISAM2 isam(parameters);

  // // Create a Factor Graph and Values to hold the new data
  // NonlinearFactorGraph graph;
  // Values initialEstimate;

  // // Loop over the poses, adding the observations to iSAM incrementally
  // for (size_t i = 0; i < poses.size(); ++i) {
  //   // Add factors for each landmark observation
  //   for (size_t j = 0; j < points.size(); ++j) {
  //     PinholeCamera<Cal3_S2> camera(poses[i], *K);
  //     Point2 measurement = camera.project(points[j]);
  //     graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(measurement, measurementNoise, Symbol('x', i), Symbol('l', j), K);
  //   }

  //   // Add an initial guess for the current pose
  //   // Intentionally initialize the variables off from the ground truth
  //   static Pose3 kDeltaPose(Rot3::Rodrigues(-0.1, 0.2, 0.25),
  //                           Point3(0.05, -0.10, 0.20));
  //   initialEstimate.insert(Symbol('x', i), poses[i] * kDeltaPose);

  //   // If this is the first iteration, add a prior on the first pose to set the
  //   // coordinate frame and a prior on the first landmark to set the scale Also,
  //   // as iSAM solves incrementally, we must wait until each is observed at
  //   // least twice before adding it to iSAM.
  //   if (i == 0) {
  //     // Add a prior on pose x0, 30cm std on x,y,z and 0.1 rad on roll,pitch,yaw
  //     static auto kPosePrior = noiseModel::Diagonal::Sigmas(
  //         (Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.3))
  //             .finished());
  //     graph.addPrior(Symbol('x', 0), poses[0], kPosePrior);

  //     // Add a prior on landmark l0
  //     static auto kPointPrior = noiseModel::Isotropic::Sigma(3, 0.1);
  //     graph.addPrior(Symbol('l', 0), points[0], kPointPrior);

  //     // Add initial guesses to all observed landmarks
  //     // Intentionally initialize the variables off from the ground truth
  //     static Point3 kDeltaPoint(-0.25, 0.20, 0.15);
  //     for (size_t j = 0; j < points.size(); ++j)
  //       initialEstimate.insert<Point3>(Symbol('l', j), points[j] + kDeltaPoint);

  //   } else {
  //     // Update iSAM with the new factors
  //     isam.update(graph, initialEstimate);
  //     // Each call to iSAM2 update(*) performs one iteration of the iterative
  //     // nonlinear solver. If accuracy is desired at the expense of time,
  //     // update(*) can be called additional times to perform multiple optimizer
  //     // iterations every step.
  //     isam.update();
  //     Values currentEstimate = isam.calculateEstimate();
  //     cout << "****************************************************" << endl;
  //     cout << "Frame " << i << ": " << endl;
  //     currentEstimate.print("Current estimate: ");

  //     // Clear the factor graph and values for the next iteration
  //     graph.resize(0);
  //     initialEstimate.clear();
  //   }
  // }
}

void sync_est_and_gt(object_data_vec_t data_est, object_data_vec_t data_gt, object_est_gt_data_vec_t& data, int object_id, double dt_thresh) {
  // data.push_back(make_tuple(5, get<1>(data_est[0]), get<1>(data_est[0])));
  // now "sync" the gt and est for each object
  
  double t_gt, t_est;
  uint next_est_time_ind = 0;
  for (uint i = 0; i < data_gt.size(); i++) {
    t_gt = get<0>(data_gt[i]);
    for (uint j = next_est_time_ind; j < data_est.size(); j++) {
      t_est = get<0>(data_est[j]);
      if(dt_thresh > abs(t_gt - t_est)) {
        data.push_back(make_tuple((t_gt + t_est)/2, object_id, get<1>(data_gt[i]), get<1>(data_est[j])));
        if (object_id != 1) { // no pose data for ego robot, all zeros
          double t_diff, rot_diff; 
          calc_pose_delta(get<1>(data_gt[i]).inverse(), get<1>(data_est[j]), &t_diff, &rot_diff, true);
          cout << "time = " << t_est << ". id = " << object_id << ".  gt / est diff:  t_delta = " << t_diff << ", r_delta = " << rot_diff << " deg" << endl;
        }
        next_est_time_ind = j + 1;
        break;
      }
    }
  }
}

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
  // roll (X) pitch (Y) yaw (Z) (set Z to 0)
  Matrix3 M = R.matrix();
  double x,y,z,cx,cy,cz,sx,sy,sz;
  sy = M(0,2);
  y = asin(sy);
  cy = cos(y);
  if (abs(cy) < 0.0001) {
    cz = M(0, 0) / cy;
    cx =  M(2, 2) / cy;
    z = acos(cz);
    x = acos(cx);
    sx = sin(x);

    // set z = 0...
    sz = 0;
    cz = 1;
    return Rot3();
  }
  else{
    cz = M(0, 0) / cy;
    cx =  M(2, 2) / cy;
    z = acos(cz);
    x = acos(cx);
    sx = sin(x);

    // set z = 0...
    sz = 0;
    cz = 1;

    // Rot3 (double R11, double R12, double R13, double R21, double R22, double R23, double R31, double R32, double R33)
    return Rot3(     cy*cz,            -cy*sz,         sy,
                cx*sz + cz*sx*sy, cx*cz - sx*sy*sz, -cy*sx,
                sx*sz - cx*cz*sy, cz*sx + cx*sy*sz,  cx*cy );
  }
  runtime_error("SHOULD NEVER BE HERE (remove_yaw)");
  return Rot3(); 

  // This has been "tested" by taking copying a matrix to matlab, checking its rotm2eul, 
  //  then comparing that to the rotm2eul of the same matrix after its gone through this function

// Per Matlab (eul2rotm function)...
  //  case 'XYZ'
  //       %     The rotation matrix R can be constructed as follows by
  //       %     ct = [cx cy cz] and st = [sx sy sz]
  //       %
  //       %     R = [            cy*cz,           -cy*sz,     sy]
  //       %         [ cx*sz + cz*sx*sy, cx*cz - sx*sy*sz, -cy*sx]
  //       %         [ sx*sz - cx*cz*sy, cz*sx + cx*sy*sz,  cx*cy]
  //       %       = Rx(tx) * Ry(ty) * Rz(tz)

}