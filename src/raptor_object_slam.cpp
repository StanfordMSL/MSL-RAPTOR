
#include "raptor_object_slam.h"

using namespace std;
using namespace gtsam;
class MSLRaptorSlamClass {
  // Note: tf_A_B is a transform that when right multipled by a vector in frame B produces the same vector in frame A: p_A = tf_A_B * p_B

  ros::NodeHandle nh;
  bool b_batch_slam;
  bool b_nocs_data;
  string ego_ns;
  map<string, obj_param_t> obj_param_map = {
    {"bowl_white_small_norm", obj_param_t("bowl_white_small_norm", "bowl",    2, false, false, false)}, //true
    {"camera_canon_len_norm", obj_param_t("camera_canon_len_norm", "camera",  3, false, false, false)},
    {"can_arizona_tea_norm",  obj_param_t("can_arizona_tea_norm",  "can",     4, false, false, false)}, //true
    {"laptop_air_xin_norm",   obj_param_t("laptop_air_xin_norm",   "laptop",  5, false, false, false)},
    {"mug_daniel_norm",       obj_param_t("mug_daniel_norm",       "cup",     6, false, false, false)},
    {"quad4",                 obj_param_t("quad4",                 "mslquad", 7, false, false, true)}, //true,
    {"quad6",                 obj_param_t("quad6",                 "mslquad", 8, false, false, true)} //true
  };

  double dt_thresh = 0.02; // how close a measurement is in time to ego pose to be "from" there - eventually should interpolate instead
  int num_ado_objs;

  public:
    MSLRaptorSlamClass(bool b_batch_slam_, string ego_ns_, bool b_nocs_data_) {
      b_batch_slam = b_batch_slam_;
      ego_ns = ego_ns_;
      b_nocs_data = b_nocs_data_;
      ROS_INFO("ego ns: %s", ego_ns.c_str());
      ROS_INFO("batch slam?   %d", b_batch_slam);
    }


    void run_batch_slam_from_rosbag(string processed_rosbag) {
      set<double> times;
      object_data_vec_t ego_data_gt, ego_data_est;
      // std::map<std::string, object_data_vec_t> ado_data_gt, ado_data_est;
      // map<string, object_est_gt_data_vec_t> raptor_data;
      vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, pair<gtsam::Pose3, gtsam::Pose3> > > > raptor_data; // time, ego_pose_gt, ego_pose_est, ado_name_to_gt_est_poses
      rslam_utils::load_rosbag(raptor_data, num_ado_objs, processed_rosbag, ego_ns, obj_param_map, dt_thresh, b_nocs_data); // "fills in" raptor_data, num_ado_objs
      string fn = "/mounted_folder/test_graphs_gtsam/batch_input1.csv";
      run_batch_slam(raptor_data, num_ado_objs);
    }


    void run_batch_slam(const vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, pair<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data, int num_ado_objs) {
      // note: object id serves as landmark id, landmark is same as saying "ado"
      // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample.cpp <-- example VO, but using betweenFactors instead of stereo


      // SET PARAMTERS FOR RUN
      bool b_use_gt = false;
      bool b_use_gt_init = false;
      if (b_use_gt) {
        b_use_gt_init = true;
      }

      // DECLARE VARIABLES 
      // These will store the following poses: ground truth, quasi-odometry, and post-slam optimized
      map<Symbol, Pose3> tf_w_gt_map, tf_w_est_preslam_map, tf_w_est_postslam_map; // these are all tf_w_ego or tf_w_ado frames. Note: gt relative pose at t0 is the same as world pose (since we make our coordinate system based on our initial ego pose)
      map<Symbol, double> ego_time_map; // store the time at each camera position
      map<Symbol, map<Symbol, pair<Pose3, Pose3> > > tf_ego_ado_maps;
      vector<pair<Pose3, Pose3>> tf_w_ego_gt_est_vec;
      int t_ind = 0, ego_pose_index = 1;

      // STEP 0) Create graph & value objects, create noise model, and add first pose at origin
      int t_ind_cutoff = 3000;
      NonlinearFactorGraph graph;
      Values initial_estimate; // create Values object to contain initial estimates of camera poses and landmark locations

      // Eventually I will use the ukf's covarience here, but for now use a constant one
      noiseModel::Diagonal::shared_ptr constNoiseMatrix;
      constNoiseMatrix = noiseModel::Diagonal::Sigmas( (Vector(6) << Vector3::Constant(.005), Vector3::Constant(.05) ).finished());

      // Add first pose ("anchor" the graph)
      bool b_landmarks_observed = false; // set to true if we observe at least 1 landmark (so we know if we should try to estimate a pose)
      Symbol ego_sym = Symbol('x', ego_pose_index);
      Pose3 first_pose = get<2>(raptor_data[0]);  // use gt value for first ego pose
      graph.emplace_shared<NonlinearEquality<Pose3> >(Symbol('x', ego_pose_index), first_pose, 1); // put a unary factor on first pose to "anchor" the whole graph

      // STEP 1) loop through ego poses, at each time do the following:
      //  - 1A) Add factors to graph between this pose and any visible landmarks various landmarks as we go 
      //  - 1B) If first timestep, use the fact that x1 is defined to be origin to set ground truth pose of landmarks. Also use first estimate as initial estimate for landmark pose
      //  - 1C) Otherwise, use gt / estimate of landmark poses from t0 and current time to gt / estimate of current camera pose (SUBSTITUE FOR ODOMETRY!!)
      //  - 1D) Use estimate of current camera pose for value initalization

      /////////////////////////////////////
      // For each ado object (landmark) there will be a single node in the graph. 
      //    Add it to the graph here and initialized it. Also save values
      map<string, gtsam::Pose3> tf_w_ado0_gt, tf_w_ado0_est;
      rslam_utils::get_tf_w_ado_for_all_objects(raptor_data, tf_w_ado0_gt, tf_w_ado0_est, num_ado_objs);

      for (auto const & key_val : tf_w_ado0_gt) {
        cout << "ado_name : " << key_val.first << "\ntf_w_ado0_gt = " << key_val.second << "tf_w_ado0_est = " << tf_w_ado0_est[key_val.first] << endl;
      }

      for (const auto & key_val : tf_w_ado0_gt) {
        string ado_name = key_val.first;
        Pose3 tf_w_ado_gt = key_val.second;
        Pose3 tf_w_ado_est = tf_w_ado0_est[ado_name];
        Symbol ado_sym = Symbol('l', obj_param_map[ado_name].obj_id);

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
            tf_w_est_preslam_map[ado_sym] = Pose3(tf_w_ado_est); 
            initial_estimate.insert(ado_sym, rslam_utils::add_noise_to_pose3(tf_w_ado_est, 0.05, 0)); 
          }
        }
      }

      for (const auto & rstep : raptor_data ) {
        double time = get<0>(rstep);
        ego_pose_index = 1 + t_ind;
        ego_sym = Symbol('x', ego_pose_index);
        gtsam::Pose3 tf_w_ego_gt  = get<1>(rstep);
        gtsam::Pose3 tf_w_ego_est = get<2>(rstep);
        map<string, pair<gtsam::Pose3, gtsam::Pose3> > measurements = get<3>(rstep);

        // initialize estimate for ego pose
        if (b_use_gt) {
          initial_estimate.insert(ego_sym, Pose3(tf_w_ego_gt));
          tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_gt);
        }
        else {
          if (b_use_gt_init) {
            initial_estimate.insert(ego_sym, Pose3(tf_w_ego_gt));
            tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_gt);
          }
          else {
            initial_estimate.insert(ego_sym, Pose3(tf_w_ego_est));
            tf_w_est_preslam_map[ego_sym] = rslam_utils::add_noise_to_pose3(tf_w_ego_est, 0.1, 0);
          }
        }

        // record values for later easy access by symbol
        ego_time_map[ego_sym] = time;
        tf_w_gt_map[ego_sym] = Pose3(tf_w_ego_gt);

        gtsam::Pose3 tf_w_ado_gt, tf_w_ado_est;
        for (const auto & key_val : measurements) {
          string ado_name = key_val.first;
          Symbol ado_sym = Symbol('l', obj_param_map[ado_name].obj_id);
          pair<gtsam::Pose3, gtsam::Pose3> ado_w_gt_est_pair = key_val.second;
          gtsam::Pose3 tf_ego_ado_gt  = tf_w_ego_gt.inverse()  * ado_w_gt_est_pair.first;  // (tf_w_ego_gt.inverse()) * tf_w_ado_gt;
          gtsam::Pose3 tf_ego_ado_est = tf_w_ego_est.inverse() * ado_w_gt_est_pair.second; // (tf_w_ego_est.inverse()) * tf_w_ado_est;
          tf_w_ado_gt  = tf_w_ego_gt  * tf_ego_ado_gt;
          tf_w_ado_est = tf_w_ego_est * tf_ego_ado_est;
          
          // record values for later easy access by symbol
          tf_ego_ado_maps[ego_sym][ado_sym] = make_pair(Pose3(tf_ego_ado_gt), Pose3(tf_ego_ado_est)); // this is so these can be written to a csv file later
          
          // Add measurement to graph
          if (b_use_gt) {
            graph.emplace_shared<BetweenFactor<Pose3> >(ego_sym, ado_sym, Pose3(tf_ego_ado_gt), constNoiseMatrix);
            // cout << ego_sym << " <--> " << ado_sym << endl;
          }
          else {
            graph.emplace_shared<BetweenFactor<Pose3> >(ego_sym, ado_sym, Pose3(tf_ego_ado_est), constNoiseMatrix);
          }
        }
        t_ind++;

        // string ado_name = get<1>(rstep);
        // Symbol ado_sym = Symbol('l', obj_param_map[ado_name].obj_id);
        // Pose3 tf_ego_ado_gt  = (get<2>(rstep).inverse()) * get<4>(rstep); // (tf_w_ego_gt.inverse()) * tf_w_ado_gt;
        // Pose3 tf_ego_ado_est = (get<3>(rstep).inverse()) * get<5>(rstep); // (tf_w_ego_est.inverse()) * tf_w_ado_est;
        // Pose3 tf_w_ego_gt  = tf_w_ado0_gt[ado_name]  * (tf_ego_ado_gt.inverse());
        // Pose3 tf_w_ego_est = tf_w_ado0_est[ado_name] * (tf_ego_ado_est.inverse());
        // Pose3 tf_w_ado_gt  = tf_w_ado0_gt[ado_name];
        // Pose3 tf_w_ado_est = tf_w_ado0_est[ado_name];


        // ego_pose_index = 1 + t_ind;
        // ego_sym = Symbol('x', ego_pose_index);

        // record values for later easy access by symbol
        // ego_time_map[ego_sym] = time;
        // tf_w_gt_map[ego_sym] = Pose3(tf_w_ego_gt);
        // tf_ego_ado_maps[ego_sym][ado_sym] = make_pair(Pose3(tf_ego_ado_gt), Pose3(tf_ego_ado_est)); // this is so these can be written to a csv file later

        // if (b_use_gt) {
        //   graph.emplace_shared<BetweenFactor<Pose3> >(ego_sym, ado_sym, Pose3(tf_ego_ado_gt), constNoiseMatrix);
        // }
        // else {
        //   graph.emplace_shared<BetweenFactor<Pose3> >(ego_sym, ado_sym, Pose3(tf_ego_ado_est), constNoiseMatrix);
        // }

        // // initialize estimate for ego pose
        // if (b_use_gt) {
        //   initial_estimate.insert(ego_sym, Pose3(tf_w_ego_gt));
        //   tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_gt);
        // }
        // else {
        //   if (b_use_gt_init) {
        //     initial_estimate.insert(ego_sym, Pose3(tf_w_ego_gt));
        //     tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_gt);
        //   }
        //   else {
        //     initial_estimate.insert(ego_sym, Pose3(tf_w_ego_est));
        //     tf_w_est_preslam_map[ego_sym] = Pose3(tf_w_ego_est);
        //   }
        // }
        // t_ind++;
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
      rslam_utils::write_results_csv(fn, raptor_data, tf_w_est_preslam_map, tf_w_est_postslam_map, tf_ego_ado_maps, obj_param_map);
    }

    ~MSLRaptorSlamClass() {}
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "raptor_object_slam");
  ros::NodeHandle nh("~");
  
  bool b_batch_slam, b_nocs_data;
  int b_batch_slam_int, b_nocs_data_int;
  nh.param("b_batch_slam", b_batch_slam_int, 1);
  b_batch_slam = b_batch_slam_int == 1;
  nh.param("b_nocs_data", b_nocs_data_int, 0);  
  b_nocs_data = b_nocs_data_int == 1;

  string ego_ns;
  nh.param<string>("ego_ns", ego_ns, "quad7");
  string input_rosbag, processed_rosbag;
  nh.param<string>("input_rosbag", input_rosbag, "");
  nh.param<string>("processed_rosbag", processed_rosbag, "");

  MSLRaptorSlamClass rslam = MSLRaptorSlamClass(b_batch_slam, ego_ns, b_nocs_data);

  rslam.run_batch_slam_from_rosbag(processed_rosbag);

  ros::Rate loop_rate(5);
  while (ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
  ros::shutdown();
  return 0;
}
