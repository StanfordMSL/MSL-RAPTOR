#include "file_io_utils.h"

namespace rslam_utils {

void load_raptor_output_rosbag(std::string rosbag_fn) {
    ROS_INFO("loading rosbag: %s",rosbag_fn.c_str());
    return;
}

void load_gt_rosbag(std::string rosbag_fn, std::string ego_ns, map<std::string, obj_param_t> obj_param_map) {
    ROS_INFO("GT - loading rosbag: %s", rosbag_fn.c_str());
    rosbag::Bag bag;
    bag.open(rosbag_fn, rosbag::bagmode::Read);

    // msl_raptor::AngledBbox cust_msg;
    std::string gt_pose_topic  = "/mavros/vision_pose/pose";
    std::string est_pose_topic = "/mavros/local_position/pose";

    std::vector<std::string> ado_names, ado_gt_topic_strs, ado_est_topic_strs;
    std::map<std::string, std::string> ado_topic_to_name;
    for (const auto & key_val : obj_param_map) {
        std::string ado_name = key_val.first;
        ado_names.push_back(ado_name);
        ado_gt_topic_strs.push_back(ado_name + gt_pose_topic);
        ado_est_topic_strs.push_back(ado_name + est_pose_topic);
        ado_topic_to_name[ado_name + gt_pose_topic] = ado_name;
        ado_topic_to_name[ado_name + est_pose_topic] = ado_name;
    }
    std::string ego_gt_topic_str = "/" + ego_ns + gt_pose_topic;
    std::string ego_est_topic_str = "/" + ego_ns + est_pose_topic;

    object_data_vec_t ego_data_gt, ego_data_est;
    std::map<std::string, object_data_vec_t> ado_data_gt, ado_data_est;

    int num_msg_total = 0, num_poses = 0;
    double time = 0.0, time0 = -1, ave_dt = 0, last_time = 0;
    geometry_msgs::PoseStamped::ConstPtr geo_msg = nullptr;
    for(rosbag::MessageInstance const m: rosbag::View(bag)) {
        if(time0 < 0) {
            time0 = m.getTime().toSec();
            time = 0.0;
        }
        else {
            last_time = time;
            time = m.getTime().toSec() - time0;
            ave_dt += time - last_time;
        }

        if (m.getTopic() == ego_gt_topic_str || ("/" + m.getTopic() == ego_gt_topic_str)) {
            geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
            if (geo_msg != nullptr) {
                ego_data_gt.push_back(make_tuple(time, ros_geo_pose_to_gtsam_pose3(geo_msg->pose)));
                num_poses++;
            }
        }
        else if (m.getTopic() == ego_est_topic_str || ("/" + m.getTopic() == ego_est_topic_str)) {
            geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
            if (geo_msg != nullptr) {
                ego_data_est.push_back(make_tuple(time, ros_geo_pose_to_gtsam_pose3(geo_msg->pose)));
                num_poses++;
            }
        }
        else {
            bool b_found = false;
            for (const auto &  topic_str : ado_gt_topic_strs) {
                if (m.getTopic() == topic_str || ("/" + m.getTopic() == topic_str)) {
                    geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
                    if (geo_msg != nullptr) {
                        ado_data_gt[ado_topic_to_name[topic_str]].push_back(make_tuple(time, ros_geo_pose_to_gtsam_pose3(geo_msg->pose)));
                        b_found = true;
                        num_poses++;
                        break;
                    }
                }
            }
            if (!b_found) {
                for (const auto &  topic_str : ado_est_topic_strs) {
                    if (m.getTopic() == topic_str || ("/" + m.getTopic() == topic_str)) {
                        geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
                        if (geo_msg != nullptr) {
                            ado_data_est[ado_topic_to_name[topic_str]].push_back(make_tuple(time, ros_geo_pose_to_gtsam_pose3(geo_msg->pose)));
                            b_found = true;
                            num_poses++;
                            break;
                        }
                    }
                }
            }
        }
        num_msg_total++;
    }
    std::cout << "num messages = " << num_msg_total << " (" << num_poses << " poses)" << std::endl;
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



    // std::string bowl_pose_est = "bowl_white_small_norm/mavros/local_position/pose";
    // std::string bowl_pose_gt = "bowl_white_small_norm/mavros/vision_pose/pose";
    // std::string camera_pose_est = "camera_canon_len_norm/mavros/local_position/pose";
    // std::string camera_pose_gt = "camera_canon_len_norm/mavros/vision_pose/pose";
    // std::string can_pose_est = "can_arizona_tea_norm/mavros/local_position/pose";
    // std::string can_pose_gt = "can_arizona_tea_norm/mavros/vision_pose/pose";
    // std::string laptop_pose_est = "laptop_air_xin_norm/mavros/local_position/pose";
    // std::string laptop_pose_gt = "laptop_air_xin_norm/mavros/vision_pose/pose";
    // std::string mug_pose_est = "mug_daniel_norm/mavros/local_position/pose";
    // std::string mug_pose_gt = "mug_daniel_norm/mavros/vision_pose/pose";
    // std::string cam_info = "/quad7/camera/camera_info";
    // std::string img = "/quad7/camera/image_raw";
    // std::string ego_pose_est = "/quad7/mavros/local_position/pose";
    // std::string ego_pose_gt = "/quad7/mavros/vision_pose/pose";