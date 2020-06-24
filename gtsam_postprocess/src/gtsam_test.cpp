/**
 * @file gtsam_test.cpp
 * @brief Attempt to read in rosbag and process with isam
 * @author Adam Caccavale
 */

/**
 * Attempt to read in a rosbag produced by MSL-RAPTOR and further process it with gtsam
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

// Includes for Reading Rosbag - http://wiki.ros.org/rosbag/Code%20API
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <tf/tfMessage.h>

// Misc Includes
#include <algorithm>
#include <random>

using namespace std;
using namespace gtsam;

// Type Defs for loading rosbag data
typedef vector<tuple<double, geometry_msgs::Pose>> object_data_vec_t;  // vector of tuples(double, ros pose message)
typedef tuple<double, int, geometry_msgs::Pose, geometry_msgs::Pose> data_tuple; 
typedef vector<data_tuple> object_est_gt_data_vec_t; //vector of tuples(double, int (id), ros pose message, ros pose message)

// Rosbg Loading & Processing Functions
void preprocess_rosbag(string bag_name, object_est_gt_data_vec_t& ego_data, object_est_gt_data_vec_t& obj_data, map<std::string, int> &object_id_map, double dt_thresh);
void sync_est_and_gt(object_data_vec_t data_est, object_data_vec_t data_gt, object_est_gt_data_vec_t& ego_data, int object_id, double dt_thresh);

// GTSAM-RAPTOR Function
void run_isam(object_est_gt_data_vec_t& all_data, map<std::string, int> &object_id_map);
void run_batch_slam(const object_est_gt_data_vec_t& ego_data, const object_est_gt_data_vec_t& obj_data, map<std::string, int> &object_id_map, double dt_thresh);

// Helper functions
void add_init_est_noise(Pose3 &ego_pose_est);
Pose3 ros_geo_pose_to_gtsam_pose3(geometry_msgs::Pose ros_pose);


int main(int argc, char** argv) {
  // useful gtsam examples:
  // https://github.com/borglab/gtsam/blob/develop/examples/VisualISAM2Example.cpp
  // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample.cpp
  // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample_large.cpp

  string bag_name = "/mounted_folder/nocs/test/scene_1.bag"; // Rosbag location & name
  double dt_thresh = 0.02; // how close a measurement is in time to ego pose to be "from" there - eventually should interpolate instead
  map<std::string, int> object_id_map = {
        {"ego", 1},
        {"bowl", 2},
        {"camera", 3},
        {"can", 4},
        {"laptop", 5},
        {"mug", 6}
  };
  
  object_est_gt_data_vec_t obj_data, ego_data;
  preprocess_rosbag(bag_name, ego_data, obj_data, object_id_map, dt_thresh);
  std::sort(obj_data.begin(), obj_data.end(), [](const data_tuple& lhs, const data_tuple& rhs) {
      return get<0>(lhs) < get<0>(rhs);
   });   // this should sort the vector by time (i.e. first element in each tuple)

  run_batch_slam(ego_data, obj_data, object_id_map, dt_thresh);
  // run_isam(all_data, object_id_map);

  cout << "done!" << endl;
  return 0;
}

void run_batch_slam(const object_est_gt_data_vec_t& ego_data, const object_est_gt_data_vec_t& obj_data, map<std::string, int> &object_id_map, double dt_thresh) {
  // note: object id serves as landmark id
  // https://github.com/borglab/gtsam/blob/develop/examples/StereoVOExample.cpp <-- example VO, but using betweenFactors instead of stereo


  // create graph object, add first pose at origin with key '1'
  NonlinearFactorGraph graph;
  Pose3 first_pose;
  int ego_pose_index = 1;
  graph.emplace_shared<NonlinearEquality<Pose3> >(Symbol('x', ego_pose_index), Pose3());


  // create Values object to contain initial estimates of camera poses and landmark locations
  Values initial_estimate;

  // Eventually I will use the ukf's covarience here, but for now use a constant one
  auto constNoiseMatrix = noiseModel::Diagonal::Sigmas(
          (Vector(6) << Vector3::Constant(0.01), Vector3::Constant(0.03)).finished());

  int obj_list_ind = 0;
  map<int, Pose3> tf_w_ado_map; // note: gt relative pose at t0 is the same as world pose (since we make our coordinate system based on our initial ego pose)
  for(int t_ind = 0; t_ind < ego_data.size(); t_ind++) {
    // loop through ego poses, adding factors to various landmarks as we go
    ego_pose_index = 1 + t_ind;

    double ego_time = get<0>(ego_data[t_ind]);
    geometry_msgs::Pose ego_est = get<3>(ego_data[t_ind]);

    if (obj_list_ind >= obj_data.size()) {
      cout << "encorperated all observations into graph" << endl;
      break;
    }

    Pose3 tf_w_ego_gt, tf_ego_ado_gt;
    
    if(t_ind > 10)
      break;
    cout << "\n-----------------------------------------" << endl;
    cout << "t_ind = " << t_ind << endl;

    while(obj_list_ind < obj_data.size() && abs(get<0>(obj_data[obj_list_ind]) - ego_time) < dt_thresh ) {
      // this means this object's measurement is from this ego pose index
      int obj_id = get<1>(obj_data[obj_list_ind]);
      Pose3 tf_ego_ado_est = ros_geo_pose_to_gtsam_pose3(get<3>(obj_data[obj_list_ind]));
      graph.emplace_shared<BetweenFactor<Pose3> >(Symbol('x', ego_pose_index), Symbol('l', obj_id), tf_ego_ado_est, constNoiseMatrix);

      if(t_ind == 0) {
        // if first loop, assume all objects are seen and store their gt values - this will be used for intializing pose estimates
        tf_w_ado_map[obj_id] = ros_geo_pose_to_gtsam_pose3(get<2>(obj_data[obj_list_ind])); // tf_w_ado
        // cout << "Object Inital Pose (tf_w_ado): " << obj_id << "\n" << tf_w_ado_map[obj_id] << endl;

        // add initial estimate for landmark
        initial_estimate.insert(Symbol('l', obj_id), tf_ego_ado_est); // since by construction at t=0 the world and ego pose are both the origin, this is valid
        tf_w_ego_gt = Pose3();
      }
      else {
        // use gt position of landmark now & at t0 to get gt position of ego. add noise to get initial pose estimate
        // cout << "Object Inital Pose again (tf_w_ado): " << obj_id << "\n" << tf_w_ado_map[obj_id] << endl; 
        tf_ego_ado_gt = ros_geo_pose_to_gtsam_pose3(get<2>(obj_data[obj_list_ind])); // current relative gt object pose

        // cout << "tf_w_ado (obj_id " << obj_id << "): " << tf_w_ado_map[obj_id] << endl;
        // cout << "tf_ego_ado_gt (obj_id " << obj_id << "): " << tf_ego_ado_gt << endl;
        tf_w_ego_gt = tf_w_ado_map[obj_id] * tf_ego_ado_gt.inverse(); // gt ego pose in world frame
        // cout << "tf_w_ego_gt (obj_id " << obj_id << "): " << tf_w_ego_gt << endl;
      }
      obj_list_ind++;
    }
    // add initial estimate for just added ego pose
    add_init_est_noise(tf_w_ego_gt);
    initial_estimate.insert(Symbol('x', ego_pose_index), tf_w_ego_gt);
    cout << "tf_w_ego_gt: " << tf_w_ego_gt << endl;

    // cout << "tmp" << endl;
  }
  cout << "done building batch slam graph!" << endl;

  // create Levenberg-Marquardt optimizer for resulting factor graph, optimize
  LevenbergMarquardtOptimizer optimizer(graph, initial_estimate);
  Values result = optimizer.optimize();

  result.print("Final result:\n");
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

void preprocess_rosbag(string bag_name, object_est_gt_data_vec_t& ego_data, object_est_gt_data_vec_t& obj_data, map<std::string, int> &object_id_map, double dt_thresh) {
  rosbag::Bag bag;
  bag.open(bag_name);  // BagMode is Read by default
  string tf = "/tf";
  string bowl_pose_est = "bowl_white_small_norm/mavros/local_position/pose";
  string bowl_pose_gt = "bowl_white_small_norm/mavros/vision_pose/pose";
  string camera_pose_est = "camera_canon_len_norm/mavros/local_position/pose";
  string camera_pose_gt = "camera_canon_len_norm/mavros/vision_pose/pose";
  string can_pose_est = "can_arizona_tea_norm/mavros/local_position/pose";
  string can_pose_gt = "can_arizona_tea_norm/mavros/vision_pose/pose";
  string laptop_pose_est = "laptop_air_xin_norm/mavros/local_position/pose";
  string laptop_pose_gt = "laptop_air_xin_norm/mavros/vision_pose/pose";
  string mug_pose_est = "mug_daniel_norm/mavros/local_position/pose";
  string mug_pose_gt = "mug_daniel_norm/mavros/vision_pose/pose";
  string cam_info = "/quad7/camera/camera_info";
  string img = "/quad7/camera/image_raw";
  string ego_pose_est = "/quad7/mavros/local_position/pose";
  string ego_pose_gt = "/quad7/mavros/vision_pose/pose";


  tf::tfMessage::ConstPtr tf_msg = nullptr;
  geometry_msgs::PoseStamped::ConstPtr geo_msg = nullptr;
  sensor_msgs::CameraInfo::ConstPtr cam_info_msg = nullptr;
  sensor_msgs::Image::ConstPtr img_msg = nullptr;
  double time = 0.0, time0 = -1, ave_dt = 0, last_time = 0;

  object_data_vec_t ego_data_est, ego_data_gt, bowl_data_est, bowl_data_gt, camera_data_est, camera_data_gt, can_data_est, can_data_gt, laptop_data_est, laptop_data_gt, mug_data_est, mug_data_gt;

  int num_msgs = 0;
  for(rosbag::MessageInstance const m: rosbag::View(bag))
  {
    // std_msgs::Int32::ConstPtr i = m.instantiate<std_msgs::Int32>();
    num_msgs++;
    if(time0 < 0) {
      time0 = m.getTime().toSec();
      time = 0.0;
    }
    else {
      last_time = time;
      time = m.getTime().toSec() - time0;
      ave_dt += time - last_time;
    }

    if (m.getTopic() == tf || ("/" + m.getTopic() == tf)) {
      tf_msg = m.instantiate<tf::tfMessage>();
      if (tf_msg != nullptr) {
        // cout << tf_msg->transforms << endl;
      }
    }
    else if (m.getTopic() == bowl_pose_est || ("/" + m.getTopic() == bowl_pose_est)) {
      geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (geo_msg != nullptr) {
        bowl_data_est.push_back(make_tuple(time, geo_msg->pose));
      }
    }
    else if (m.getTopic() == bowl_pose_gt || ("/" + m.getTopic() == bowl_pose_gt)) {
      geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (geo_msg != nullptr) {
        bowl_data_gt.push_back(make_tuple(time, geo_msg->pose));
      }
    }
    else if (m.getTopic() == camera_pose_est || ("/" + m.getTopic() == camera_pose_est)) {
      geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (geo_msg != nullptr) {
        camera_data_est.push_back(make_tuple(time, geo_msg->pose));
      }
    }
    else if (m.getTopic() == camera_pose_gt || ("/" + m.getTopic() == camera_pose_gt)) {
      geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (geo_msg != nullptr) {
        camera_data_gt.push_back(make_tuple(time, geo_msg->pose));
      }
    }
    else if (m.getTopic() == can_pose_est || ("/" + m.getTopic() == can_pose_est)) {
      geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (geo_msg != nullptr) {
        can_data_est.push_back(make_tuple(time, geo_msg->pose));
      }
    }
    else if (m.getTopic() == can_pose_gt || ("/" + m.getTopic() == can_pose_gt)) {
      geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (geo_msg != nullptr) {
        can_data_gt.push_back(make_tuple(time, geo_msg->pose));
      }
    }
    else if (m.getTopic() == laptop_pose_est || ("/" + m.getTopic() == laptop_pose_est)) {
      geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (geo_msg != nullptr) {
        laptop_data_est.push_back(make_tuple(time, geo_msg->pose));
      }
    }
    else if (m.getTopic() == laptop_pose_gt || ("/" + m.getTopic() == laptop_pose_gt)) {
      geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (geo_msg != nullptr) {
        laptop_data_gt.push_back(make_tuple(time, geo_msg->pose));
      }
    }
    else if (m.getTopic() == mug_pose_est || ("/" + m.getTopic() == mug_pose_est)) {
      geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (geo_msg != nullptr) {
        mug_data_est.push_back(make_tuple(time, geo_msg->pose));
        // cout << get<1>(mug_data_est.back()) << endl;
      }
    }
    else if (m.getTopic() == mug_pose_gt || ("/" + m.getTopic() == mug_pose_gt)) {
      geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (geo_msg != nullptr) {
        mug_data_gt.push_back(make_tuple(time, geo_msg->pose));
      }
    }
    else if (m.getTopic() == cam_info || ("/" + m.getTopic() == cam_info)) {
      if(cam_info_msg != nullptr) {
        continue;
      }
      cam_info_msg = m.instantiate<sensor_msgs::CameraInfo>();
    }
    else if (m.getTopic() == img || ("/" + m.getTopic() == img)) {
      img_msg = m.instantiate<sensor_msgs::Image>();
      if (img_msg != nullptr) {
        // cout << img_msg->pose << endl;
      }
    }
    else if (m.getTopic() == ego_pose_est || ("/" + m.getTopic() == ego_pose_est)) {
      geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (geo_msg != nullptr) {
        ego_data_est.push_back(make_tuple(time, geo_msg->pose));
      }
    }
    else if (m.getTopic() == ego_pose_gt || ("/" + m.getTopic() == ego_pose_gt)) {
      geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
      if (geo_msg != nullptr) {
        ego_data_gt.push_back(make_tuple(time, geo_msg->pose));
      }
    }
    else {
      cout << "Unexpected message type found. Topic: " << m.getTopic() << " Type: " << m.getDataType() << endl;
    }
  }
  ave_dt /= num_msgs - 1;
  cout << "Number of messages in bag = " << num_msgs << endl;
  cout << "Average timestep = " << ave_dt << endl;
  bag.close();

  // object_est_gt_data_vec_t* ego_data = sync_est_and_gt(ego_data_est, ego_data_gt);
  // &all_data[0] = &ego_data;


  object_est_gt_data_vec_t bowl_data, camera_data, can_data, laptop_data, mug_data;

  sync_est_and_gt(ego_data_est, ego_data_gt, ego_data, object_id_map["ego"], dt_thresh);

  sync_est_and_gt(bowl_data_est, bowl_data_gt, bowl_data, object_id_map["bowl"], dt_thresh);
  obj_data.insert( obj_data.end(), bowl_data.begin(), bowl_data.end() );

  sync_est_and_gt(camera_data_est, camera_data_gt, camera_data, object_id_map["camera"], dt_thresh);
  obj_data.insert( obj_data.end(), camera_data.begin(), camera_data.end() );

  sync_est_and_gt(can_data_est, can_data_gt, can_data, object_id_map["can"], dt_thresh);
  obj_data.insert( obj_data.end(), can_data.begin(), can_data.end() );

  sync_est_and_gt(laptop_data_est, laptop_data_gt, laptop_data, object_id_map["laptop"], dt_thresh);
  obj_data.insert( obj_data.end(), laptop_data.begin(), laptop_data.end() );

  sync_est_and_gt(mug_data_est, mug_data_gt, mug_data, object_id_map["mug"], dt_thresh);
  obj_data.insert( obj_data.end(), mug_data.begin(), mug_data.end() );
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
        next_est_time_ind = j + 1;
        break;
      }
    }
  }
  // return &data;
}

void add_init_est_noise(Pose3 &ego_pose_est) {
  // noise = np.array([random.uniform(-0.02, 0.02) for i in range(3)]) 
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-0.00005, 0.00005);
  // Pose3 delta(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(dis(gen), dis(gen), dis(gen)));
  Pose3 delta(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(0.00,0.00,0.00));
  cout << "noise:" << delta << "tf before noise: " << ego_pose_est << endl;
  ego_pose_est = ego_pose_est.compose(delta);
  cout << "tf after noise: " << ego_pose_est << endl;

}

Pose3 ros_geo_pose_to_gtsam_pose3(geometry_msgs::Pose ros_pose) {
  // Convert a ros pose structure to gtsam's Pose3 class
  Point3 t = Point3(ros_pose.position.x, ros_pose.position.y, ros_pose.position.z);
  Rot3 R = Rot3( Quaternion(ros_pose.orientation.w, 
                            ros_pose.orientation.x, 
                            ros_pose.orientation.y, 
                            ros_pose.orientation.z) );
  return Pose3(R, t);
}