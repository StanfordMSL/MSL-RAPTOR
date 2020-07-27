
#include "raptor_object_slam.h"

using namespace std;
using namespace gtsam;
class MSLRaptorSlamClass {
  // Note: tf_A_B is a transform that when right multipled by a vector in frame B produces the same vector in frame A: p_A = tf_A_B * p_B

  ros::NodeHandle nh;
  bool b_batch_slam;
  string ego_ns;
  map<string, obj_param_t> obj_param_map = {
    {"bowl_white_small_norm", obj_param_t("bowl_white_small_norm", "bowl",   2, false, false, false)}, //true
    {"camera_canon_len_norm", obj_param_t("camera_canon_len_norm", "camera", 3, false, false, false)},
    {"can_arizona_tea_norm",  obj_param_t("can_arizona_tea_norm",  "can",    4, false, false, false)}, //true
    {"laptop_air_xin_norm",   obj_param_t("laptop_air_xin_norm",   "laptop", 5, false, false, false)},
    {"mug_daniel_norm",       obj_param_t("mug_daniel_norm",       "cup",    6, false, false, false)}
  };

  double dt_thresh = 0.02; // how close a measurement is in time to ego pose to be "from" there - eventually should interpolate instead
  // set<double> times;
  // object_data_vec_t ego_data_gt;
  // std::map<std::string, object_data_vec_t> ado_data_gt;
  // object_est_gt_data_vec_t obj_data; // vector<data_tuple>
  // data_tuple a;

  public:
    MSLRaptorSlamClass(bool b_batch_slam_, string ego_ns_) {
      b_batch_slam = b_batch_slam_;
      ego_ns = ego_ns_;
      ROS_INFO("ego ns: %s", ego_ns.c_str());
      ROS_INFO("batch slam?   %d", b_batch_slam);

      // dt_thresh = 0.02; // how close a measurement is in time to ego pose to be "from" there - eventually should interpolate instead
      
      string path = "/mounted_folder/nocs_logs/";
      string base = "log_1_";

      // object_est_gt_data_vec_t ado_data; // all ado data in 1 vector sorted by time (to be filled in by load_log_files)
      // set<double> times;  // set of all unique times (to be filled in by load_log_files)
      // rslam_utils::load_log_files(times, ado_data, path, base, obj_param_map, dt_thresh);
      
      // if (b_batch_slam) {
      //   rslam_utils::load_raptor_output_rosbag(5);
        // run_batch_slam(times, ado_data, obj_param_map, dt_thresh);
      // }
      
    }


    void run_batch_slam_from_rosbag(string processed_rosbag) {
      set<double> times;
      object_data_vec_t ego_data_gt, ego_data_est;
      // std::map<std::string, object_data_vec_t> ado_data_gt, ado_data_est;
      // map<string, object_est_gt_data_vec_t> raptor_data;
      vector<tuple<double, Pose3, Pose3, map<string, tuple<Pose3, Pose3> > > > raptor_data; // time, ego_pose_gt, ego_pose_est, ado_name_to_gt_est_poses
      rslam_utils::load_rosbag(raptor_data, processed_rosbag, ego_ns, obj_param_map, dt_thresh); // "fills in" times, ego_data_gt, ado_data_gt
      run_batch_slam(raptor_data);
    }


    // void run_batch_slam() {
    void run_batch_slam(vector<tuple<double, Pose3, Pose3, map<string, tuple<Pose3, Pose3> > > > raptor_data) {
      // note: object id serves as landmark id, landmark is same as saying "ado"
      // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample.cpp <-- example VO, but using betweenFactors instead of stereo
      // map<double, Pose3> tf_w_ego_gt_map;
      // map<Symbol, map<double, Pose3> > tf_w_ado_gt_map;

      // STEP 0) Create graph & value objects, add first pose at origin
      int t_ind_cutoff = 3000;
      NonlinearFactorGraph graph;
      int ego_pose_index = 1;
      Values initial_estimate; // create Values object to contain initial estimates of camera poses and landmark locations

      // bool b_fake_perfect_measurements = true;
      bool b_use_gt = false;
      bool b_use_gt_init = true;
      if (b_use_gt) {
        b_use_gt_init = true;
      }

      // Eventually I will use the ukf's covarience here, but for now use a constant one
      noiseModel::Diagonal::shared_ptr constNoiseMatrix;
      constNoiseMatrix = noiseModel::Diagonal::Sigmas( (Vector(6) << Vector3::Constant(.005), Vector3::Constant(.05) ).finished());

      
      // These will store the following poses: ground truth, quasi-odometry, and post-slam optimized
      map<Symbol, Pose3> tf_w_gt_map, tf_w_est_preslam_map, tf_w_est_postslam_map; // these are all tf_w_ego or tf_w_ado frames. Note: gt relative pose at t0 is the same as world pose (since we make our coordinate system based on our initial ego pose)
      map<Symbol, double> ego_time_map; // store the time at each camera position
      map<Symbol, map<Symbol, pair<Pose3, Pose3> > > tf_ego_ado_maps;

      vector<pair<Pose3, Pose3>> tf_w_ego_gt_est_vec;
      // STEP 1) loop through ego poses, at each time do the following:
      //  - 1A) Add factors to graph between this pose and any visible landmarks various landmarks as we go 
      //  - 1B) If first timestep, use the fact that x1 is defined to be origin to set ground truth pose of landmarks. Also use first estimate as initial estimate for landmark pose
      //  - 1C) Otherwise, use gt / estimate of landmark poses from t0 and current time to gt / estimate of current camera pose (SUBSTITUE FOR ODOMETRY!!)
      //  - 1D) Use estimate of current camera pose for value initalization
      int t_ind = 0;
      bool b_landmarks_observed = false; // set to true if we observe at least 1 landmark (so we know if we should try to estimate a pose)
      Symbol ego_sym = Symbol('x', ego_pose_index);
      Pose3 first_pose = get<1>(raptor_data[0]);  // use gt value for first ego pose
      graph.emplace_shared<NonlinearEquality<Pose3> >(Symbol('x', ego_pose_index), first_pose); // put a unary factor on first pose to "anchor" the whole graph
      for (const auto & raptor_step : raptor_data ) {
        double time = get<0>(raptor_step);
        Pose3 tf_w_ego_gt = get<1>(raptor_step);
        Pose3 tf_w_ego_est = get<2>(raptor_step);
        map<string, tuple<Pose3, Pose3> > measurements = get<3>(raptor_step);

        ego_pose_index = 1 + t_ind;
        ego_sym = Symbol('x', ego_pose_index);

        for (const auto & single_ado_meas : measurements) {
          string ado_name = single_ado_meas.first;
          Pose3 tf_w_ado_gt = get<0>(single_ado_meas.second);
          Pose3 tf_w_ado_est = get<1>(single_ado_meas.second);
          cout << "double check frames for ado_est!!!" << endl;
          Symbol ado_sym = Symbol('l', obj_param_map[ado_name].obj_id);

          if(tf_w_gt_map.find(ado_sym) == tf_w_gt_map.end()) {
            // we have not seen this object yet
            // - save ado world pose
            // - initial estimate for ado pose
            tf_w_gt_map[ado_sym] = tf_w_ado_gt;
            if (b_use_gt) {
              tf_w_est_preslam_map[ado_sym] = Pose3(tf_w_ado_gt);
              initial_estimate.insert(ado_sym, Pose3(tf_w_ado_gt));
            }
            else {
              if (b_use_gt_init) {
                tf_w_est_preslam_map[ado_sym] = Pose3(tf_w_ado_gt);
                initial_estimate.insert(ado_sym, Pose3(tf_w_ado_gt)); 
              }
              else {
                tf_w_est_preslam_map[ado_sym] = Pose3(tf_w_ado_est); //add_noise_to_pose3(tf_w_ado_gt, .02, .06); 
                initial_estimate.insert(ado_sym, Pose3(tf_w_ado_est)); 
              }
            }
          }
          // Any time including first:
          // - add relative pose to graph
          // - initial estimate for ego pose
          Pose3 tf_ego_ado_gt, tf_ego_ado_est;
          tf_ego_ado_gt = (tf_w_ego_gt.inverse()) * tf_w_ado_gt;
          tf_ego_ado_est = (tf_w_ego_est.inverse()) * tf_w_ado_est;
          // tf_ego_ado_gt  = get<2>(obj_data[obj_list_ind]); // current relative gt object pose
          // tf_ego_ado_est = get<2>(obj_data[obj_list_ind]); // estimated ado pose
          // tf_ego_ado_est = get<3>(obj_data[obj_list_ind]); // estimated ado pose

          tf_ego_ado_maps[ego_sym][ado_sym] = make_pair(Pose3(tf_ego_ado_gt), Pose3(tf_ego_ado_est)); // this is so these can be written to a csv file later

          // 1A) - add ego pose <--> landmark (i.e. ado) pose factor. syntax is: ego_id ("x1"), ado_id("l3"), measurment (i.e. relative pose in ego frame tf_ego_ado_est), measurement uncertanty (covarience)
          if (b_use_gt) {
            graph.emplace_shared<BetweenFactor<Pose3> >(ego_sym, ado_sym, Pose3(tf_ego_ado_gt), constNoiseMatrix);
          }
          else {
            graph.emplace_shared<BetweenFactor<Pose3> >(ego_sym, ado_sym, Pose3(tf_ego_ado_est), constNoiseMatrix);
          }

          // 1C) use gt position of landmark now & at t0 to get gt position of ego. Same for estimated position
          tf_w_ego_est = tf_w_est_preslam_map[ado_sym] * tf_ego_ado_est.inverse(); // est ego pose in world frame
          // if (b_use_gt) {
          //   tf_w_ego_est = Pose3(tf_w_ego_gt); // DEBUG ONLY!!!!
          // }
        }

        // 1D) only calculate our pose if we actually see objects
        if (b_use_gt) {
          initial_estimate.insert(ego_sym, Pose3(tf_w_ego_gt));
          tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_gt);
          // initial_estimate.insert(ego_sym, Pose3(tf_w_ego_gt));
          // tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_gt);
        }
        else {
          if (b_use_gt_init) {
            initial_estimate.insert(ego_sym, Pose3(tf_w_ego_gt));
            tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_gt);
          }
          else {
            initial_estimate.insert(ego_sym, Pose3(tf_w_ego_est));
            tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_est);
          }
        }
        tf_w_gt_map[ego_sym] = Pose3(tf_w_ego_gt);
        ego_time_map[ego_sym] = time;
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
        rslam_utils::calc_pose_delta(tf_w_est_preslam, tf_w_gt_inv, &t_diff_pre_val, &rot_diff_pre_val, b_degrees);
        rslam_utils::calc_pose_delta(tf_w_est_postslam, tf_w_gt_inv, &t_diff_post_val, &rot_diff_post_val, b_degrees);
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
      rslam_utils::write_results_csv(fn, ego_time_map, tf_w_gt_map, tf_w_est_preslam_map, tf_w_est_postslam_map, tf_ego_ado_maps);
    }

    // void load_gt(string rosbag_fn) {
    //   rslam_utils::load_gt_rosbag(times, ego_data_gt, ado_data_gt, rosbag_fn, ego_ns, obj_param_map, dt_thresh);
    // }
    // void load_raptor_data(string rosbag_fn) {
    //   rslam_utils::load_raptor_output_rosbag(rosbag_fn, ego_ns, obj_param_map);
    // }

    ~MSLRaptorSlamClass() {}
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "raptor_object_slam");
  ros::NodeHandle nh("~");

  ros::Rate loop_rate(5);
  bool b_batch_slam;
  // ros::param::get("~batch", strtmp);
  nh.param<bool>("batch_slam", b_batch_slam, true);
  string ego_ns;
  nh.param<string>("ego_ns", ego_ns, "quad7");
  string input_rosbag, processed_rosbag;
  nh.param<string>("input_rosbag", input_rosbag, "");
  nh.param<string>("processed_rosbag", processed_rosbag, "");

  // string my_test_string = "... this is a test...\n";
  MSLRaptorSlamClass rslam = MSLRaptorSlamClass(b_batch_slam, ego_ns);

  rslam.run_batch_slam_from_rosbag(processed_rosbag);

  /**
   * A count of how many messages we have sent. This is used to create
   * a unique string for each message.
   */
  int count = 0;
  while (ros::ok())
  {
    std_msgs::String msg;
    stringstream ss;
    ss << "hello raptor object slam world " << count;
    msg.data = ss.str();
    // ROS_INFO("%s", msg.data.c_str());
    // chatter_pub.publish(msg);

    ros::spinOnce();

    loop_rate.sleep();
    ++count;
  }
  ros::shutdown();
  return 0;
}
