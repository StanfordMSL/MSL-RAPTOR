#include "file_io_utils.h"

namespace rslam_utils {

void load_raptor_output_rosbag(std::string rosbag_fn) {
    ROS_INFO("loading rosbag: %s",rosbag_fn.c_str());
    return;
}

void load_gt_rosbag(std::string rosbag_fn) {
    ROS_INFO("GT - loading rosbag: %s", rosbag_fn.c_str());
    rosbag::Bag bag;
    bag.open(rosbag_fn, rosbag::bagmode::Read);

    msl_raptor::AngledBbox cust_msg;
    
    int i = 0;
    for(rosbag::MessageInstance const m: rosbag::View(bag)) {
    //   std_msgs::Int32::ConstPtr i = m.instantiate<std_msgs::Int32>();
    //   if (i != nullptr)
    //     std::cout << i->data << std::endl;
        // if (m.getTopic() == r_cam_image || ("/" + m.getTopic() == r_cam_image))
        // {
        //     sensor_msgs::Image::ConstPtr r_img = m.instantiate<sensor_msgs::Image>();
        //     if (r_img != NULL)
        //         r_img_sub.newMessage(r_img);
        // }
        
        i++;
    }
    std::cout << "num messages = " << i << std::endl;
    bag.close();
    return;
}

}