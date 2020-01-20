#!/usr/bin/env python3
# IMPORTS
# system
import sys, time
from copy import copy
import pdb
# math
import numpy as np
from scipy.spatial.transform import Rotation as R
# plots
import matplotlib
matplotlib.use('Agg')  ## This is needed for the gui to work from a virtual container
import matplotlib.pyplot as plt
# ros
import rosbag
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError

class result_analyser:

    def __init__(self, rb_name=None, quad_ns="quad4"):
        if rb_name is not None:
            if len(rb_name) > 4 and rb_name[-4:] == ".bag":
                self.rb_name = rb_name
            else:
                self.rb_name = rb_name + ".bag"
        else:
            raise RuntimeError("rb_name = None - need to provide rosbag name (with or without .bag)")

        try:
            self.bag = rosbag.Bag(self.rb_name, 'r')
        except Exception as e:
            raise RuntimeError("Unable to Process Rosbag!!\n{}".format(e))

        self.ado_gt_topic = quad_ns + '/mavros/vision_pose/pose'
        self.ado_est_topic = quad_ns + '/msl_raptor_state'

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
        self.init_plot()
        self.process_rb()


    def process_rb(self):
        for topic, msg, t in self.bag.read_messages(topics=[self.ado_gt_topic, self.ado_est_topic]):
            if topic == self.ado_gt_topic:
                self.parse_ado_gt_msg(msg, t)
            elif topic == self.ado_est_topic:
                self.parse_ado_est_msg(msg, t)
        self.update_plot()


    def parse_ado_est_msg(self, msg, t=None):
        """
        record estimated poses from msl-raptor
        """
        if t is None:
            self.t_est.append(msg.header.stamp.to_sec())
        else:
            self.t_est.append(t)

        my_state = self.pose_to_state_vec(msg.pose)
        rpy = self.quat_to_ang(np.reshape(my_state[6:10], (1,4)))[0]
        self.x_est.append(my_state[0])
        self.y_est.append(my_state[1])
        self.z_est.append(my_state[2])
        self.roll_est.append(rpy[0])
        self.pitch_est.append(rpy[1])
        self.yaw_est.append(rpy[2])


    def parse_ado_gt_msg(self, msg, t=None):
        """
        record optitrack poses of tracked quad
        """
        if t is None:
            self.t_gt.append(msg.header.stamp.to_sec())
        else:
            self.t_gt.append(t)

        my_state = self.pose_to_state_vec(msg.pose)
        rpy = self.quat_to_ang(np.reshape(my_state[6:10], (1,4)))[0]
        self.x_gt.append(my_state[0])
        self.y_gt.append(my_state[1])
        self.z_gt.append(my_state[2])
        self.roll_gt.append(rpy[0])
        self.pitch_gt.append(rpy[1])
        self.yaw_gt.append(rpy[2])


    def quat_to_ang(self, q):
        """
        Convert a quaternion to euler angles (ASSUMES 'XYZ')
        """
        return R.from_quat(np.roll(q,3,axis=1)).as_euler('XYZ')


    def pose_to_state_vec(self, pose):
        """ Turn a ROS pose message into a 13 el state vector (w/ 0 vels) """
        state = np.zeros((13,))
        state[0:3] = [pose.position.x, pose.position.y, pose.position.z]
        state[6:10] = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        if state[6] < 0:
            state[6:10] *= -1
        return state


    def init_plot(self):
        self.fig, self.axes = plt.subplots(3, 2, clear=True)
        est_line_style = 'r-'
        gt_line_style = 'b-'
        self.x_gt_plt, = self.axes[0,0].plot(0, 0, gt_line_style)
        self.x_est_plt, = self.axes[0,0].plot(0, 0, est_line_style)
        self.y_gt_plt, = self.axes[1,0].plot(0, 0, gt_line_style)
        self.y_est_plt, = self.axes[1,0].plot(0, 0, est_line_style)
        self.z_gt_plt, = self.axes[2,0].plot(0, 0, gt_line_style)
        self.z_est_plt, = self.axes[2,0].plot(0, 0, est_line_style)
        self.roll_gt_plt, = self.axes[0,1].plot(0, 0, gt_line_style)
        self.roll_est_plt, = self.axes[0,1].plot(0, 0, est_line_style)
        self.pitch_gt_plt, = self.axes[1,1].plot(0, 0, gt_line_style)
        self.pitch_est_plt, = self.axes[1,1].plot(0, 0, est_line_style)
        self.yaw_gt_plt, = self.axes[2,1].plot(0, 0, gt_line_style)
        self.yaw_est_plt, = self.axes[2,1].plot(0, 0, est_line_style)
        plt.show(block=False)

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

        plt.draw()
        plt.pause(0.0001)


if __name__ == '__main__':
    try:
        program = result_analyser()
    except:
        import traceback
        traceback.print_exc()

