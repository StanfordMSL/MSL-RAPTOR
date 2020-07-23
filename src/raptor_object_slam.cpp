
#include "raptor_object_slam.h"


class MSLRaptorSlamClass {
  ros::NodeHandle nh;
  bool b_batch_slam;


  public:
    MSLRaptorSlamClass(bool b_batch_slam_, std::string test_string) {
      ROS_INFO("IN CLASS: %s", test_string.c_str());
      b_batch_slam = b_batch_slam_;
      ROS_INFO("batch slam?   %d", b_batch_slam);
      gtsam::Pose3 pose_test;
      rosbag::Bag bag;
      gtsam::Values initial_estimate;
      gtsam::utilities::extractPose3(initial_estimate);


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

      // object_est_gt_data_vec_t ado_data; // all ado data in 1 vector sorted by time (to be filled in by load_log_files)
      set<double> times;  // set of all unique times (to be filled in by load_log_files)
      // load_log_files(times, ado_data, path, base, obj_param_map, dt_thresh);
      
      // if (b_batch_slam) {
      //   rslam_utils::load_raptor_output_rosbag(5);
      //   run_batch_slam(times, ado_data, obj_param_map, dt_thresh);
      // }
      

    }

    void load_gt(std::string rosbag_fn) {
      rslam_utils::load_gt_rosbag(rosbag_fn);
    }

    ~MSLRaptorSlamClass() {
    }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "gtsam_ros_test");
  ros::NodeHandle nh("~");

  ros::Rate loop_rate(5);
  bool b_batch_slam;
  // ros::param::get("~batch", strtmp);
  nh.param<bool>("batch_slam", b_batch_slam, true);
  std::string input_rosbag, processed_rosbag;
  nh.param<std::string>("input_rosbag", input_rosbag, "");
  nh.param<std::string>("processed_rosbag", processed_rosbag, "");

  // std::string my_test_string = "... this is a test...\n";
  MSLRaptorSlamClass rslam = MSLRaptorSlamClass(b_batch_slam, "... this is a test...\n");

  rslam.load_gt(input_rosbag);

  /**
   * A count of how many messages we have sent. This is used to create
   * a unique string for each message.
   */
  int count = 0;
  while (ros::ok())
  {
    std_msgs::String msg;
    std::stringstream ss;
    ss << "hello raptor object slam world " << count;
    msg.data = ss.str();
    ROS_INFO("%s", msg.data.c_str());
    // chatter_pub.publish(msg);

    ros::spinOnce();

    loop_rate.sleep();
    ++count;
  }
  ros::shutdown();
  return 0;
}
