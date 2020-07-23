
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
    }

    ~MSLRaptorSlamClass() {
    }
};

int main(int argc, char **argv)
{
  /**
   * The ros::init() function needs to see argc and argv so that it can perform
   * any ROS arguments and name remapping that were provided at the command line.
   * For programmatic remappings you can use a different version of init() which takes
   * remappings directly, but for most command-line programs, passing argc and argv is
   * the easiest way to do it.  The third argument to init() is the name of the node.
   *
   * You must call one of the versions of ros::init() before using any other
   * part of the ROS system.
   */
  ros::init(argc, argv, "gtsam_ros_test");
  // ros::NodeHandle nh;
  ros::NodeHandle nh("~");
  // ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 20);

  ros::Rate loop_rate(5);
  bool b_batch_slam;
  // ros::param::get("~batch", strtmp);
  nh.param<bool>("batch_slam", b_batch_slam, true);

  // std::string my_test_string = "... this is a test...\n";
  MSLRaptorSlamClass(b_batch_slam, "... this is a test...\n");

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
