#!/usr/bin/env python3

# IMPORTS
# system
import os, sys, argparse, time
import pdb
# from pathlib import Path
# save/load
# import pickle
# math
import numpy as np
# plots
# import matplotlib
# from matplotlib import pyplot as plt
# from mpl_toolkits import mplot3d
# ros
import rospy
# custom modules
from ros_interface import ros_interface as ROS
from ukf import UKF
# libs & utils
from utils_msl_raptor.ros_utils import *
from utils_msl_raptor.math_utils import *
# sys.path.append('/root/msl_raptor_ws/src/msl_raptor/src/front_end/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src/front_end')
from image_segmentor import ImageSegmentor
# IMPORTS
# system
import sys, time
import pdb
# math
import numpy as np
# ros
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf
# libs & utils
from utils_msl_raptor.ros_utils import *
# from utils_msl_raptor.ukf_utils import *
import cv2
from cv_bridge import CvBridge, CvBridgeError

class bb_viz_node:

    def __init__(self):
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
        

    def image_cb(self, msg):
        """
        Maintains a buffer of images & times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """

        my_time = get_ros_time()  # time in seconds

        if len(self.img_buffer[0]) < self.img_rosmesg_buffer_len:
            self.img_buffer[0].append(msg.pose)
            self.img_buffer[1].append(my_time)
        else:
            self.img_buffer[0][0:self.img_rosmesg_buffer_len] = self.img_buffer[0][1:self.img_rosmesg_buffer_len]
            self.img_buffer[1][0:self.img_rosmesg_buffer_len] = self.img_buffer[1][1:self.img_rosmesg_buffer_len]
            self.img_buffer[0][-1] = msg.pose
            self.img_buffer[1][-1] = my_time


    def bb_viz_cb(self, msg):
        my_time = msg.data[-1]
        im_seg_mode = msg.data[-2]
        bb_data = msg.data[0:-2]
        im_msg = find_closest_by_time(my_time, self.ego_pose_rosmesg_buffer[1], self.ego_pose_rosmesg_buffer[0])[0]
        image = self.bridge.imgmsg_to_cv2(im_msg, desired_encoding="passthrough")

        cv2.drawContours(image,[bb_data],0,(0,191,255),2)

        self.image_bb_pub.publish(self.bridge.cv2_to_imgmsg(image, "passthrough"))


    def run(self):
        rate = rospy.Rate(15)

        while not rospy.is_shutdown():
            self.itr += 1
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

