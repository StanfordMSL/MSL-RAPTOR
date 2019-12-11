#!/usr/bin/env python3
# IMPORTS
# system
import sys, time
import pdb
# math
import numpy as np
from bisect import bisect_left
# ros
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf
# libs & utils
# from utils_msl_raptor.ros_utils import *
# from utils_msl_raptor.ukf_utils import *
import cv2
from cv_bridge import CvBridge, CvBridgeError

class bb_viz_node:

    def __init__(self):
        rospy.init_node('bb_viz_node', anonymous=True)
        self.DETECT = 1
        self.TRACK = 2
        self.REINIT = 3
        self.IGNORE = 4

        self.bridge = CvBridge()
        self.img_buffer = ([], [])
        self.img_rosmesg_buffer_len = 10

        self.ns = rospy.get_param('~ns')  # robot namespace
        self.overlaid_img = None

        rospy.Subscriber(self.ns + '/camera/image_raw', Image, self.image_cb, queue_size=1,buff_size=2**21)
        self.bb_data_sub = rospy.Subscriber(self.ns + '/bb_data', Float32MultiArray, self.bb_viz_cb, queue_size=5)
        self.img_overlay_pub = rospy.Publisher(self.ns + '/image_bb_overlay', Image, queue_size=5)
        self.itr = 0
        

    def image_cb(self, msg):
        """
        Maintains a buffer of images & times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """

        my_time = rospy.Time.now().to_sec()  # time in seconds

        if len(self.img_buffer[0]) < self.img_rosmesg_buffer_len:
            self.img_buffer[0].append(msg)
            self.img_buffer[1].append(my_time)
        else:
            self.img_buffer[0][0:self.img_rosmesg_buffer_len] = self.img_buffer[0][1:self.img_rosmesg_buffer_len]
            self.img_buffer[1][0:self.img_rosmesg_buffer_len] = self.img_buffer[1][1:self.img_rosmesg_buffer_len]
            self.img_buffer[0][-1] = msg
            self.img_buffer[1][-1] = my_time


    def find_closest_by_time_ros2(self, time_to_match, time_list, message_list=None):
        """
        Assumes lists are sorted earlier to later. Returns closest item in list by time. If two numbers are equally close, return the smallest number.
        Adapted from https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
        """
        if not message_list:
            message_list = time_list
        pos = bisect_left(time_list, time_to_match)
        if pos == 0:
            return message_list[0], 0
        if pos == len(time_list):
            return message_list[-1], len(message_list) - 1
        before = time_list[pos - 1]
        after = time_list[pos]
        if after - time_to_match < time_to_match - before:
            return message_list[pos], pos
        else:
            return message_list[pos - 1], pos - 1


    def bb_viz_cb(self, msg):
        if not self.img_buffer or len(self.img_buffer[0]) == 0:
            return
        my_time = msg.data[-1]
        im_seg_mode = msg.data[-2]
        bb_data = msg.data[0:-2]
        im_msg = self.find_closest_by_time_ros2(my_time, self.img_buffer[1], self.img_buffer[0])[0]
        image = self.bridge.imgmsg_to_cv2(im_msg, desired_encoding="passthrough")
        box = np.int0(cv2.boxPoints( ( (bb_data[1], bb_data[0]), (bb_data[2], bb_data[3]), -np.degrees(bb_data[4]))) )
        cv2.drawContours(image,[box],0,(0,255,0),2)
        print(bb_data)
        self.img_overlay_pub.publish(self.bridge.cv2_to_imgmsg(image, "passthrough"))
        # cv2.imwrite('/test_img{}.png'.format(self.itr), image)
        self.itr += 1
        # pdb.set_trace()


    def run(self):
        rate = rospy.Rate(15)

        while not rospy.is_shutdown():
            if self.overlaid_img is not None:
                self.img_overlay_pub.publish(self.overlaid_img)
            rate.sleep()


if __name__ == '__main__':
    try:
        program = bb_viz_node()
        program.run()
    except:
        import traceback
        traceback.print_exc()

