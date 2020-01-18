#!/usr/bin/env python3
# IMPORTS
# system
import sys, time
from copy import copy
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
from utils_msl_raptor.ros_utils import *
from utils_msl_raptor.math_utils import *
import cv2
from cv_bridge import CvBridge, CvBridgeError

class plot_6dof:

    def __init__(self):
        rospy.init_node('plot_6dof_node', anonymous=True)
        self.fig = None
        self.axes = None
        self.t_est = []
        self.t_gt = []
        self.x_est = []
        self.x_gt = []
        self.y_est = []
        self.y_gt = []
        self.z_est = []
        self.z_gt = []
        self.roll_est = []
        self.roll_gt = []
        self.pitch_est = []
        self.pitch_gt = []
        self.yaw_est = []
        self.yaw_gt = []
        self.tracked_quad_ns = rospy.get_param('~tracked_quad_ns')
        rospy.Subscriber(self.tracked_quad_ns + '/msl_raptor_state', PoseStamped, queue_size=5, self.ado_pose_cb, queue_size=10)  # msl-raptor estimate of pose
        rospy.Subscriber(self.tracked_quad_ns + '/mavros/vision_pose/pose', PoseStamped, self.ado_pose_gt_cb, queue_size=10)  # optitrack gt pose
        self.init_plot()


    def ado_pose_cb(self, msg):
        """
        record estimated poses from msl-raptor
        """
        self.t_est.append(get_ros_time(msg))
        my_state = pose_to_state_vec(msg)
        my_roll, my_pitch, my_yaw = quat_to_ang(my_state[6:10])
        self.x_est.append(my_state[0])
        self.y_est.append(my_state[1])
        self.z_est.append(my_state[2])
        self.roll_est.append(my_roll)
        self.pitch_est.append(my_pitch)
        self.yaw_est.append(my_yaw)


    def ado_pose_gt_cb(self, msg):
        """
        record optitrack poses of tracked quad
        """
        self.t_gt.append(get_ros_time(msg))
        my_state = pose_to_state_vec(msg)
        my_roll, my_pitch, my_yaw = quat_to_ang(my_state[6:10])
        self.x_gt.append(my_state[0])
        self.y_gt.append(my_state[1])
        self.z_gt.append(my_state[2])
        self.roll_gt.append(my_roll)
        self.pitch_gt.append(my_pitch)
        self.yaw_gt.append(my_yaw)


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

    def init_plot(self):
        self.fig, self.axes = plt.subplots(3, 2, clear=True)  # sharex=True,
        est_line_style = 'r-'
        gt_line_style = 'b-'
        self.x_gt_plt, = self.axes[0,0].plot(None, None, gt_line_style)
        self.x_est_plt, = self.axes[0,0].plot(None, None, est_line_style)
        self.y_gt_plt, = self.axes[0,0].plot(None, None, gt_line_style)
        self.y_est_plt, = self.axes[0,0].plot(None, None, est_line_style)
        self.z_gt_plt, = self.axes[0,0].plot(None, None, gt_line_style)
        self.z_est_plt, = self.axes[0,0].plot(None, None, est_line_style)
        self.roll_gt_plt, = self.axes[0,0].plot(None, None, gt_line_style)
        self.roll_est_plt, = self.axes[0,0].plot(None, None, est_line_style)
        self.pitch_gt_plt, = self.axes[0,0].plot(None, None, gt_line_style)
        self.pitch_est_plt, = self.axes[0,0].plot(None, None, est_line_style)
        self.yaw_gt_plt, = self.axes[0,0].plot(None, None, gt_line_style)
        self.yaw_est_plt, = self.axes[0,0].plot(None, None, est_line_style)

    def update_plot(self):
        self.x_gt_plt.set_xdata(self.t_gt)
        self.x_est_plt.set_xdata(self.t_est)
        self.x_gt_plt.set_ydata(self.x_gt)
        self.x_est_plt.set_ydata(self.x_est)

        self.y_gt_plt.set_xdata(self.t_gt)
        self.y_est_plt.set_xdata(self.t_est)
        self.y_gt_plt.set_ydata(self.y_gt)
        self.y_est_plt.set_ydata(self.y_est)

        self.z_gt_plt.set_xdata(self.t_gt)
        self.z_est_plt.set_xdata(self.t_est)
        self.z_gt_plt.set_ydata(self.x_gt)
        self.z_est_plt.set_ydata(self.x_est)

        self.roll_gt_plt.set_xdata(self.t_gt)
        self.roll_est_plt.set_xdata(self.t_est)
        self.roll_gt_plt.set_ydata(self.roll_gt)
        self.roll_est_plt.set_ydata(self.roll_est)

        self.pitch_gt_plt.set_xdata(self.t_gt)
        self.pitch_est_plt.set_xdata(self.t_est)
        self.pitch_gt_plt.set_ydata(self.pitch_gt)
        self.pitch_est_plt.set_ydata(self.pitch_est)

        self.yaw_gt_plt.set_xdata(self.t_gt)
        self.yaw_est_plt.set_xdata(self.t_est)
        self.yaw_gt_plt.set_ydata(self.yaw_gt)
        self.yaw_est_plt.set_ydata(self.yaw_est)


    def run(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if len(self.t_est) > 0:
                self.update_plot()
            rate.sleep()


if __name__ == '__main__':
    try:
        program = plot_6dof()
        program.run()
    except:
        import traceback
        traceback.print_exc()

