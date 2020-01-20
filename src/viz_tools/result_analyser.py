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
# matplotlib.use('Agg')  ## This is needed for the gui to work from a virtual container
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

    def __init__(self, rb_name=None, ego_quad_ns="/quad7", ado_quad_ns="/quad4"):
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

        self.ado_gt_topic = ado_quad_ns + '/mavros/vision_pose/pose'
        self.ado_est_topic = ego_quad_ns + '/msl_raptor_state'  # ego_quad_ns since it is ego_quad's estimate of the ado quad

        self.fig = None
        self.axes = None
        self.t0 = -1
        self.tf = -1
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
        self.process_rb()


    def process_rb(self):
        print("Processing {}".format(self.rb_name))
        for topic, msg, t in self.bag.read_messages(topics=[self.ado_gt_topic, self.ado_est_topic]):
            if topic == self.ado_gt_topic:
                self.parse_ado_gt_msg(msg, t.to_sec())
            elif topic == self.ado_est_topic:
                self.parse_ado_est_msg(msg, t.to_sec())

        self.t_est = np.asarray(self.t_est)
        self.t0 = np.min(self.t_est)
        self.tf = np.max(self.t_est) - self.t0
        self.t_est -= self.t0
        self.t_gt = np.asarray(self.t_gt) - self.t0
        pdb.set_trace()
        self.do_plot()


    def do_plot(self):
        self.fig, self.axes = plt.subplots(3, 2, clear=True)
        est_line_style = 'r-'
        gt_line_style = 'b-'
        self.x_gt_plt, = self.axes[0,0].plot(self.t_gt, self.x_gt, gt_line_style)
        self.x_est_plt, = self.axes[0,0].plot(self.t_est, self.x_est, est_line_style)
        self.y_gt_plt, = self.axes[1,0].plot(self.t_gt, self.y_gt, gt_line_style)
        self.y_est_plt, = self.axes[1,0].plot(self.t_est, self.y_est, est_line_style)
        self.z_gt_plt, = self.axes[2,0].plot(self.t_gt, self.z_gt, gt_line_style)
        self.z_est_plt, = self.axes[2,0].plot(self.t_est, self.z_est, est_line_style)
        self.roll_gt_plt, = self.axes[0,1].plot(self.t_gt, self.roll_gt, gt_line_style)
        self.roll_est_plt, = self.axes[0,1].plot(self.t_est, self.roll_est, est_line_style)
        self.pitch_gt_plt, = self.axes[1,1].plot(self.t_gt, self.pitch_gt, gt_line_style)
        self.pitch_est_plt, = self.axes[1,1].plot(self.t_est, self.pitch_est, est_line_style)
        self.yaw_gt_plt, = self.axes[2,1].plot(self.t_gt, self.yaw_gt, gt_line_style)
        self.yaw_est_plt, = self.axes[2,1].plot(self.t_est, self.yaw_est, est_line_style)
        for ax in np.reshape(self.axes, (self.axes.size)):
            ax.set_xlim([0, self.tf])
        plt.show(block=False)
       

    def parse_ado_est_msg(self, msg, t=None):
        """
        record estimated poses from msl-raptor
        """
        if t is None:
            self.t_est.append(msg.header.stamp.to_sec() - self.t0)
        else:
            self.t_est.append(t - self.t0)

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


if __name__ == '__main__':
    try:
        if len(sys.argv) == 1:
            raise RuntimeError("not enough arguments, must pass in the rosbag name (w/ or w/o .bag)")
        elif len(sys.argv) > 2:
            raise RuntimeError("too many arguments, only pass in the rosbag name (w/ or w/o .bag)")
        program = result_analyser(rb_name=sys.argv[1])
        input("Press enter to close program")
        
    except:
        import traceback
        traceback.print_exc()

