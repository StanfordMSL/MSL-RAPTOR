#include "file_io_utils.h"

namespace rslam_utils {

void load_raptor_output_rosbag(std::string rosbag_fn) {
    ROS_INFO("loading rosbag: %s",rosbag_fn.c_str());
    return;
}

void load_gt_rosbag(std::string rosbag_fn, std::string ego_ns, map<string, obj_param_t> obj_param_map) {
    ROS_INFO("GT - loading rosbag: %s", rosbag_fn.c_str());
    rosbag::Bag bag;
    bag.open(rosbag_fn, rosbag::bagmode::Read);

    // msl_raptor::AngledBbox cust_msg;
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

    geometry_msgs::PoseStamped::ConstPtr geo_msg = nullptr;
    // sensor_msgs::CameraInfo::ConstPtr cam_info_msg = nullptr;
    // std_msgs::Int32::ConstPtr tmp = nullptr;

    int i = 0;
    double time = 0.0, time0 = -1, ave_dt = 0, last_time = 0;
    for(rosbag::MessageInstance const m: rosbag::View(bag)) {
    //   std_msgs::Int32::ConstPtr i = m.instantiate<std_msgs::Int32>();
    //   if (i != nullptr)
    //     std::cout << i->data << std::endl;
        if(time0 < 0) {
            time0 = m.getTime().toSec();
            time = 0.0;
        }
        else {
            last_time = time;
            time = m.getTime().toSec() - time0;
            ave_dt += time - last_time;
        }
        i++;
    }
    std::cout << "num messages = " << i << std::endl;
    bag.close();
    return;
}

gtsam::Pose3 ros_geo_pose_to_gtsam_pose3(geometry_msgs::Pose ros_pose) {
  // Convert a ros pose structure to gtsam's Pose3 class
  gtsam::Point3 t = gtsam::Point3(ros_pose.position.x, ros_pose.position.y, ros_pose.position.z);
  gtsam::Rot3 R   = gtsam::Rot3( gtsam::Quaternion(ros_pose.orientation.w, 
                                                   ros_pose.orientation.x, 
                                                   ros_pose.orientation.y, 
                                                   ros_pose.orientation.z) );
  return gtsam::Pose3(R, t);
}

}