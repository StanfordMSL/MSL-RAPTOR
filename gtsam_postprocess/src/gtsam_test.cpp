// #include <iostream>

// int main() {
//     std::cout << "Hello World!" << std::endl;
// }

 
/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file OdometryExample.cpp
 * @brief Simple robot motion example, with prior and two odometry measurements
 * @author Frank Dellaert
 */

/**
 * Example of a simple 2D localization example
 *  - Robot poses are facing along the X axis (horizontal, to the right in 2D)
 *  - The robot moves 2 meters each step
 *  - We have full odometry between poses
 */

// We will use Pose2 variables (x, y, theta) to represent the robot positions
#include <gtsam/geometry/Pose2.h>

// In GTSAM, measurement functions are represented as 'factors'. Several common factors
// have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
// Here we will use Between factors for the relative motion described by odometry measurements.
// Also, we will initialize the robot at the origin using a Prior factor.
#include <gtsam/slam/BetweenFactor.h>

// When the factors are created, we will add them to a Factor Graph. As the factors we are using
// are nonlinear factors, we will need a Nonlinear Factor Graph.
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

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

// Reading Rosbag Includes - http://wiki.ros.org/rosbag/Code%20API
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <tf/tfMessage.h>

using namespace std;
using namespace gtsam;

typedef vector<tuple<double, geometry_msgs::Pose>> object_data_vec_t;  // vector of tuples(double, ros pose message)
typedef vector<tuple<double, geometry_msgs::Pose, geometry_msgs::Pose>> object_est_gt_data_vec_t; //vector of tuples(double, ros pose message, ros pose message)

void preprocess_rosbag(string bag_name, vector<object_est_gt_data_vec_t>& all_data);
void sync_est_and_gt(object_data_vec_t data_est, object_data_vec_t data_gt, object_est_gt_data_vec_t& ego_data);


int main(int argc, char** argv) {
  // https://github.com/borglab/gtsam/blob/develop/examples/VisualISAM2Example.cpp
  string bag_name = "/mounted_folder/nocs/test/scene_1.bag";
  
  vector<object_est_gt_data_vec_t> all_data;
  preprocess_rosbag(bag_name, all_data);

  cout << "done!" << endl;
  return 0;
}

void preprocess_rosbag(string bag_name, vector<object_est_gt_data_vec_t>& all_data) {
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
        cout << get<1>(mug_data_est.back()) << endl;
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


  object_est_gt_data_vec_t ego_data, bowl_data, camera_data, can_data, laptop_data, mug_data;
  // sync_est_and_gt(mug_data_est, mug_data_gt, mug_data);
  // cout << "test 1" <<endl;
  // cout << get<0>(mug_data[0]) <<endl;
  // cout << get<1>(mug_data[0]) <<endl;
  // cout << "test 2" <<endl;
  // all_data.push_back(mug_data);
  // cout << "test 3" <<endl;
  sync_est_and_gt(ego_data_est, ego_data_gt, ego_data);
  all_data.push_back(ego_data);
  sync_est_and_gt(bowl_data_est, bowl_data_gt, bowl_data);
  all_data.push_back(bowl_data);
  sync_est_and_gt(camera_data_est, camera_data_gt, camera_data);
  all_data.push_back(camera_data);
  sync_est_and_gt(can_data_est, can_data_gt, can_data);
  all_data.push_back(can_data);
  sync_est_and_gt(laptop_data_est, laptop_data_gt, laptop_data);
  all_data.push_back(laptop_data);
  sync_est_and_gt(mug_data_est, mug_data_gt, mug_data);
  all_data.push_back(mug_data);
  // object_est_gt_data_vec_t test_data;
  // test_data.push_back(make_tuple(5, get<1>(ego_data_gt[0]), get<1>(ego_data_gt[0])));

}

void sync_est_and_gt(object_data_vec_t data_est, object_data_vec_t data_gt, object_est_gt_data_vec_t& data) {
  // data.push_back(make_tuple(5, get<1>(data_est[0]), get<1>(data_est[0])));
  // now "sync" the gt and est for each object
  
  double dt_thresh = 0.02, t_gt, t_est;
  uint next_est_time_ind = 0;
  for (uint i = 0; i < data_gt.size(); i++) {
    t_gt = get<0>(data_gt[i]);
    for (uint j = next_est_time_ind; j < data_est.size(); j++) {
      t_est = get<0>(data_est[j]);
      if(dt_thresh > abs(t_gt - t_est)) {
        data.push_back(make_tuple((t_gt + t_est)/2, get<1>(data_gt[i]), get<1>(data_est[j])));
        next_est_time_ind = j + 1;
        break;
      }
    }
  }
  // return &data;
}