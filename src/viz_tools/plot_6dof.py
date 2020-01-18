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
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf
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
        rospy.Subscriber(self.tracked_quad_ns + '/msl_raptor_state', PoseStamped, self.ado_pose_cb, queue_size=10)  # msl-raptor estimate of pose
        rospy.Subscriber(self.tracked_quad_ns + '/mavros/vision_pose/pose', PoseStamped, self.ado_pose_gt_cb, queue_size=10)  # optitrack gt pose
        self.init_plot()


    def ado_pose_cb(self, msg):
        """
        record estimated poses from msl-raptor
        """
        self.t_est.append(msg.header.stamp.to_sec())
        my_state = self.pose_to_state_vec(msg.pose)
        rpy = self.quat_to_ang(np.reshape(my_state[6:10], (1,4)))[0]
        self.x_est.append(my_state[0])
        self.y_est.append(my_state[1])
        self.z_est.append(my_state[2])
        self.roll_est.append(rpy[0])
        self.pitch_est.append(rpy[1])
        self.yaw_est.append(rpy[2])
        self.update_plot()


    def ado_pose_gt_cb(self, msg):
        """
        record optitrack poses of tracked quad
        """
        self.t_gt.append(msg.header.stamp.to_sec())
        my_state = self.pose_to_state_vec(msg.pose)
        rpy = self.quat_to_ang(np.reshape(my_state[6:10], (1,4)))[0]
        self.x_gt.append(my_state[0])
        self.y_gt.append(my_state[1])
        self.z_gt.append(my_state[2])
        self.roll_gt.append(rpy[0])
        self.pitch_gt.append(rpy[1])
        self.yaw_gt.append(rpy[2])
        self.update_plot()


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
        self.fig, self.axes = plt.subplots(3, 2, clear=True)  # sharex=True,
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
        # plt.show()
        # plt.show(block=False)


    def run(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            # if len(self.t_est) > 0:
            #     self.update_plot()
            rate.sleep()


if __name__ == '__main__':
    try:
        program = plot_6dof()
        program.run()
    except:
        import traceback
        traceback.print_exc()

