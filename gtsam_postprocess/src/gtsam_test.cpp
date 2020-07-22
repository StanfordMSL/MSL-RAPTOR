/**
 * @file gtsam_test.cpp
 * @brief Attempt to read in data and process with isam
 * @author Adam Caccavale
 */

// My custom utils
#include "msl_raptor_gtsam_utils.h"
#include "gtsam_test.h"


int main(int argc, char** argv) {
  // useful gtsam examples:
  // https://github.com/borglab/gtsam/blob/develop/examples/VisualISAM2Example.cpp
  // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample.cpp
  // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample_large.cpp
  // Note: tf_A_B is a transform that when right multipled by a vector in frame B produces the same vector in frame A: p_A = tf_A_B * p_B

  double dt_thresh = 0.02; // how close a measurement is in time to ego pose to be "from" there - eventually should interpolate instead
  map<string, obj_param_t> obj_param_map = {
    {"bowl_white_small_norm", obj_param_t("bowl_white_small_norm", "bowl",   2, false, false, false)}, //true
    {"camera_canon_len_norm", obj_param_t("camera_canon_len_norm", "camera", 3, false, false, false)},
    {"can_arizona_tea_norm",  obj_param_t("can_arizona_tea_norm",  "can",    4, false, false, false)}, //true
    {"laptop_air_xin_norm",   obj_param_t("laptop_air_xin_norm",   "laptop", 5, false, false, false)},
    {"mug_daniel_norm",       obj_param_t("mug_daniel_norm",       "mug",    6, false, false, false)}
  };
  string path = "/mounted_folder/nocs_logs/";
  string base = "log_1_";

  object_est_gt_data_vec_t ado_data; // all ado data in 1 vector sorted by time (to be filled in by load_log_files)
  set<double> times;  // set of all unique times (to be filled in by load_log_files)
  load_log_files(times, ado_data, path, base, obj_param_map, dt_thresh);

  bool b_gen_all_trajs = false;
  if(b_gen_all_trajs) {
    map<Symbol, map<double, pair<Pose3, Pose3> > > all_trajs;
    // gen_all_fake_trajectories(all_trajs, times, obj_data, t_ind_cutoff, dt_thresh);
    load_all_trajectories(all_trajs, times, "/mounted_folder/nocs_logs/", obj_param_map, dt_thresh);
    string fn_traj = "/mounted_folder/test_graphs_gtsam/graph1.csv";
    write_all_traj_csv(fn_traj, all_trajs);
    cout << "CUT SHORT FOR TRAJ LOAD - done with main!" << endl;
    return 0;
  }
  

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
  map<Symbol, map<double, pair<Pose3, Pose3> > > all_trajs;
  set<double> tmp;
  load_all_trajectories(all_trajs, tmp, "/mounted_folder/nocs_logs/", obj_params_map, dt_thresh);
  map<double, Pose3> tf_w_ego_gt_map;
  map<Symbol, map<double, Pose3> > tf_w_ado_gt_map;

  for(const auto & ego_time : times) {
    bool b_found_matching_time = false;
    for (const auto & key_val : all_trajs) {
      Symbol sym = key_val.first;
      map<double, pair<Pose3, Pose3> > pose_map = key_val.second;
      for (const auto & key_val2 : pose_map) {
        double time = key_val2.first;
        if (abs(time - ego_time) < dt_thresh) {
          tf_w_ado_gt_map[sym][ego_time] = key_val2.second.second; // tf_w_ado_gt
          tf_w_ego_gt_map[ego_time] = key_val2.second.first; // tf_w_ego_gt
          b_found_matching_time = true;
          break;
        }
      }
      // if (b_found_matching_time) {
      //   break;
      // }
    }
  }

  // STEP 0) Create graph & value objects, add first pose at origin
  int t_ind_cutoff = 3000;
  NonlinearFactorGraph graph;
  Pose3 first_pose = Pose3();
  int ego_pose_index = 1;
  Symbol ego_sym = Symbol('x', ego_pose_index);
  graph.emplace_shared<NonlinearEquality<Pose3> >(Symbol('x', ego_pose_index), first_pose);
  Values initial_estimate; // create Values object to contain initial estimates of camera poses and landmark locations

  // bool b_fake_perfect_measurements = true;
  bool b_use_poses = true;
  bool b_fake_traj = false;
  bool b_use_gt = false;
  bool b_use_gt_init = true;
  bool b_robust = false;

  // Eventually I will use the ukf's covarience here, but for now use a constant one
  noiseModel::Diagonal::shared_ptr constNoiseMatrix;
  noiseModel::Robust::shared_ptr huber;
  if (b_use_poses) {
    constNoiseMatrix = noiseModel::Diagonal::Sigmas( (Vector(6) << Vector3::Constant(.1), Vector3::Constant(1.5) ).finished());
  }
  else {
    constNoiseMatrix = noiseModel::Diagonal::Sigmas((Vector(3)<<0.01,0.01,0.03).finished());
  }
  if (b_robust) {
    // auto gaussian = noiseModel::Isotropic::Sigma(6, 0.01);
    huber = noiseModel::Robust::Create( noiseModel::mEstimator::Huber::Create(1.345), constNoiseMatrix);
  }
  
  // These will store the following poses: ground truth, quasi-odometry, and post-slam optimized
  map<Symbol, Pose3> tf_w_gt_map, tf_w_est_preslam_map, tf_w_est_postslam_map; // these are all tf_w_ego or tf_w_ado frames. Note: gt relative pose at t0 is the same as world pose (since we make our coordinate system based on our initial ego pose)
  map<Symbol, double> ego_time_map; // store the time at each camera position
  map<Symbol, map<Symbol, pair<Pose3, Pose3> > > tf_ego_ado_maps;

  vector<pair<Pose3, Pose3>> tf_w_ego_gt_est_vec;
  if (b_fake_traj) {
    gen_fake_trajectory(tf_w_ego_gt_est_vec, times, obj_data, t_ind_cutoff, dt_thresh);
  }
  // STEP 1) loop through ego poses, at each time do the following:
  //  - 1A) Add factors to graph between this pose and any visible landmarks various landmarks as we go 
  //  - 1B) If first timestep, use the fact that x1 is defined to be origin to set ground truth pose of landmarks. Also use first estimate as initial estimate for landmark pose
  //  - 1C) Otherwise, use gt / estimate of landmark poses from t0 and current time to gt / estimate of current camera pose (SUBSTITUE FOR ODOMETRY!!)
  //  - 1D) Use estimate of current camera pose for value initalization
  int t_ind = 0, obj_list_ind = 0; // this needs to be flexible since I dont know how many landmarks I see at each time, if any
  bool b_landmarks_observed = false; // set to true if we observe at least 1 landmark (so we know if we should try to estimate a pose)
  for(const auto & ego_time : times) {
    if (t_ind > t_ind_cutoff) {
      break;
    }
    if(tf_w_ego_gt_map.find(ego_time) == tf_w_ego_gt_map.end()){
      cout << "EMPTY tf_w_ego" << endl;
      continue;
    }
    ego_pose_index = 1 + t_ind;
    ego_sym = Symbol('x', ego_pose_index);

    Pose3 tf_w_ego_gt, tf_w_ego_est;
    if (b_fake_traj) {
      tf_w_ego_gt  = tf_w_ego_gt_est_vec[t_ind].first;
      tf_w_ego_est = tf_w_ego_gt_est_vec[t_ind].second;
    }
    
    // cout << "\n-----------------------------------------" << endl;
    // cout << "t_ind = " << t_ind << endl;
    while(obj_list_ind < obj_data.size() && abs(get<0>(obj_data[obj_list_ind]) - ego_time) < dt_thresh ) {
      // DATA TYPE object_est_gt_data_vec_t: vector of tuples, each tuple is: <double time, int class id, Pose3 gt pose, Pose3 est pose>
      // first condition means we have more data to process, second means this observation is cooresponding with this ego pose
      int obj_id = get<1>(obj_data[obj_list_ind]);
      Symbol ado_sym = Symbol('l', obj_id);
      if(obj_id == 2 || obj_id == 4) {
        obj_list_ind++;
        continue;
      }

      b_landmarks_observed = true; // set to true because we have at least 1 object seen
      Pose3 tf_ego_ado_gt, tf_ego_ado_est;
      
      if(t_ind == 0) { 
        // 1B) if first loop, assume all objects are seen and store their gt values - this will be used for intializing pose estimates
        Pose3 tf_w_ado_gt  = tf_w_ado_gt_map[ado_sym][ego_time]; //get<2>(obj_data[obj_list_ind]); // tf_w_ado_gt - since by construction at t=0 the world and ego pose are both the origin, this relative measurement is also in world frame
        Pose3 tf_w_ado_est = get<3>(obj_data[obj_list_ind]); // tf_w_ado_est
        if (b_use_gt) {
          tf_w_ado_est = Pose3(tf_w_ado_gt); // DEBUG ONLY!!!!
        }
        tf_w_gt_map[ado_sym] = tf_w_ado_gt;
        // tf_w_est_preslam_map[ado_sym] = tf_w_ado_est;
        // corrupted_pose = tf_w_ado_gt.compose(Pose3(Rot3::Rodrigues(-0.1, 0.2, 0.25), Point3(0.05, -0.10, 0.20)));
        Pose3 corrupted_ado_pose;
        if (b_use_gt_init) {
          corrupted_ado_pose = Pose3(tf_w_ado_gt);
        }
        else {
          corrupted_ado_pose = Pose3(tf_w_ado_est); //add_noise_to_pose3(tf_w_ado_gt, .02, .06); 
        }
        tf_w_est_preslam_map[ado_sym] = corrupted_ado_pose;

        // double t_diff_pre_val, rot_diff_pre_val;//, t_diff_post_val, rot_diff_post_val; 
        // calc_pose_delta(tf_w_ado_gt, tf_w_ado_gt.inverse(), &t_diff_pre_val, &rot_diff_pre_val, true);
        // cout << "***delta pre-slam: t = " << t_diff_pre_val << ", ang = " << rot_diff_pre_val << " deg" << endl;

        // add initial estimate for landmark (in world frame)
        if (b_use_poses) {
          initial_estimate.insert(ado_sym, corrupted_ado_pose); 
          // initial_estimate.insert(ado_sym, tf_w_ado_est); 
          // initial_estimate.insert(ado_sym, tf_w_ado_gt); 
          // initial_estimate.insert(ado_sym, add_noise_to_pose3(tf_w_ado_gt, .005, .03)); 
        }
        else {
          initial_estimate.insert(ado_sym, Point3(tf_w_ado_est.translation())); 
        }
        if (!b_fake_traj){
          tf_w_ego_gt = Pose3();
        }
        obj_list_ind++;
        continue;
      }

      if (!b_fake_traj){
        tf_ego_ado_gt = (tf_w_ego_gt_map[ego_time].inverse()) * tf_w_ado_gt_map[ado_sym][ego_time];
        // tf_ego_ado_gt  = get<2>(obj_data[obj_list_ind]); // current relative gt object pose
        tf_ego_ado_est = get<3>(obj_data[obj_list_ind]); // estimated ado pose
      }
      else {
        tf_ego_ado_gt = tf_w_ego_gt.inverse() * tf_w_gt_map[ado_sym]; 
        tf_ego_ado_est = tf_w_ego_est.inverse() * tf_w_est_preslam_map[ado_sym]; 
        // tf_ego_ado_est = tf_ego_ado_gt.compose( Pose3(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(dis(gen), dis(gen), dis(gen))) );
        
      }
      if (b_use_gt) {
        tf_ego_ado_est = Pose3(tf_ego_ado_gt); // DEBUG ONLY!!!!
      }

      tf_ego_ado_maps[ego_sym][ado_sym] = make_pair(Pose3(tf_ego_ado_gt), Pose3(tf_ego_ado_est)); // this is so these can be written to a csv file later

      // 1A) - add ego pose <--> landmark (i.e. ado) pose factor. syntax is: ego_id ("x1"), ado_id("l3"), measurment (i.e. relative pose in ego frame tf_ego_ado_est), measurement uncertanty (covarience)
      if (b_use_poses) {
        if (!b_robust) {
          graph.emplace_shared<BetweenFactor<Pose3> >(ego_sym, ado_sym, Pose3(tf_ego_ado_est), constNoiseMatrix);
        }
        else {
          graph.emplace_shared<BetweenFactor<Pose3> >(ego_sym, ado_sym, Pose3(tf_ego_ado_est), huber);
        }
      }
      else {
        Point3 meas_vec = tf_ego_ado_est.translation();
        Unit3 bearing3d = Unit3(meas_vec);
        double range3d = meas_vec.squaredNorm();
        if (!b_robust) {
          graph.emplace_shared<BearingRangeFactor<Pose3, Point3> >(ego_sym, ado_sym, bearing3d, range3d, constNoiseMatrix);
        }
        else {
          graph.emplace_shared<BearingRangeFactor<Pose3, Point3> >(ego_sym, ado_sym, bearing3d, range3d, huber);
        }
      }

      // 1C) use gt position of landmark now & at t0 to get gt position of ego. Same for estimated position
      if (!b_fake_traj){
        // tf_w_ego_gt = tf_w_gt_map[ado_sym] * tf_ego_ado_gt.inverse(); // gt ego pose in world frame
        tf_w_ego_gt = tf_w_ego_gt_map[ego_time];
        tf_w_ego_est = tf_w_est_preslam_map[ado_sym] * tf_ego_ado_est.inverse(); // est ego pose in world frame
      }
      if (b_use_gt) {
        tf_w_ego_est = Pose3(tf_w_ego_gt); // DEBUG ONLY!!!!
      }
      obj_list_ind++;
    }
    if (b_landmarks_observed) {
      // 1D) only calculate our pose if we actually see objects
      if (!b_use_gt) {
        tf_w_ego_est = Pose3(tf_w_ego_gt); // DEBUG!!! <-- only used when i want gt initialization but not gt other things
        if (b_use_gt_init) {
          initial_estimate.insert(ego_sym, Pose3(tf_w_ego_gt));
        }
        else {
          initial_estimate.insert(ego_sym, Pose3(tf_w_ego_est));
        }
        tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_est);
      }
      else {
        Pose3 corrupted_ego_pose = tf_w_ego_gt;
        // tf_w_ego_gt = add_noise_to_pose3(tf_w_ego_gt, .005, .03); 
        // Pose3 corrupted_ego_pose = add_noise_to_pose3(tf_w_ego_gt, .00001, .00004); 
        // Pose3 corrupted_ego_pose = tf_w_ego_gt.compose(Pose3());
        // Pose3 corrupted_ego_pose = Pose3();
        // Pose3 corrupted_ego_pose = tf_w_ego_gt.compose(Pose3(Rot3::Rodrigues(0, 0, 0), Point3(1, 0.00, 0.00)));
        // cout << tf_w_ego_gt << endl;
        // Pose3 corrupted_ego_pose = tf_w_ego_gt * Pose3(Rot3::Rodrigues(0, 0, 0), Point3(0.05, 0.00, 0.00));
        // cout << corrupted_ego_pose << endl;
        initial_estimate.insert(ego_sym, Pose3(corrupted_ego_pose));
        tf_w_est_preslam_map[ego_sym] = Pose3(corrupted_ego_pose);
        // initial_estimate.insert(ego_sym, Pose3(tf_w_ego_gt));
        // tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_gt);
      }
      tf_w_gt_map[ego_sym] = Pose3(tf_w_ego_gt);
      ego_time_map[ego_sym] = ego_time;
    }
    b_landmarks_observed = false;
    t_ind++;
  }
  cout << "done building batch slam graph (w/ initializations)!" << endl;

  // STEP 2) create Levenberg-Marquardt optimizer for resulting factor graph, optimize
  LevenbergMarquardtOptimizer optimizer(graph, initial_estimate);
  Values result = optimizer.optimize();
  cout << "done optimizing pose graph" << endl;

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

  string fn = "/mounted_folder/test_graphs_gtsam/graph1.csv";

  cout << "writing results to: " << fn << endl;
  write_results_csv(fn, ego_time_map, tf_w_gt_map, tf_w_est_preslam_map, tf_w_est_postslam_map, tf_ego_ado_maps);
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
