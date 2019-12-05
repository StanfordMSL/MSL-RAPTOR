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
from utils.ros_utils import *
from utils.math_utils import *


def run_execution_loop():
    b_DEBUG = True
    if b_DEBUG:
        rospy.logwarn("\n\n\n------------- IN DEBUG MODE -------------\n\n\n")
        time.sleep(0.5)
    rate = rospy.Rate(100) # max filter rate
    b_target_in_view = True
    ros = ROS(b_DEBUG)  # create a ros interface object
    wait_intil_ros_ready(ros)  # pause to allow ros to get initial messages
    ukf = UKF()  # create ukf object
    init_objects(ros, ukf)  # init camera, pose, etc

    state_est = np.zeros((ukf.dim_state + ukf.dim_sig**2, ))
    loop_count = 0
    last_image_time = 0

    rospy.logwarn("FIXING OUR POSE!!")
    while not rospy.is_shutdown():
        # store data locally (so it doesnt get overwritten in ROS object)
        loop_time = ros.latest_time
        if loop_time <= last_image_time:
            # this means we dont have new data yet
            continue
        tf_ego_w = inv_tf(pose_to_tf(ros.pose_w_ego))
        if 0:
            abb = ros.latest_bb  # angled bounding box
        else:
            abb = ukf.predict_measurement(pose_to_state_vec(ros.tracked_quad_pose_gt), tf_ego_w)
        bb_method = ros.latest_bb_method  # 1 for detect network, -1 for tracking network

        dt = loop_time - last_image_time
        ukf.itr_time = loop_time
        ukf.step_ukf(abb, tf_ego_w, dt)  # update ukf
        last_image_time = loop_time  # this ensures we dont reuse the image
        
        ros.publish_filter_state(ukf.mu, ukf.itr_time, ukf.itr)  # send vector with time, iteration, state_est
        
        rate.sleep()
        loop_count += 1
        print(" ")  # print blank line to separate iteration output
    print("ENDED")


def init_objects(ros, ukf):
    # create camera object (see https://github.com/StanfordMSL/uav_game/blob/tro_experiments/ec_quad_sim/ec_quad_sim/param/quad3_trans.yaml)
    ukf.camera = camera(ros)

    # init ukf state
    rospy.logwarn('using ground truth to initialize filter!')
    ukf.mu = pose_to_state_vec(ros.tracked_quad_pose_gt) 
    # ukf.mu[0:3] += np.array([-2, .5, .5]) 

    # init 3d bounding box in quad frame
    half_length = rospy.get_param('~target_bound_box_l') / 2
    half_width = rospy.get_param('~target_bound_box_w') / 2
    half_height = rospy.get_param('~target_bound_box_h') / 2
    ukf.bb_3d = np.array([[ half_length, half_width, half_height, 1.],  # 1 front, left,  up (from quad's perspective)
                          [ half_length, half_width,-half_height, 1.],  # 2 front, right, up
                          [ half_length,-half_width,-half_height, 1.],  # 3 back,  right, up
                          [ half_length,-half_width, half_height, 1.],  # 4 back,  left,  up
                          [-half_length,-half_width, half_height, 1.],  # 5 front, left,  down
                          [-half_length,-half_width,-half_height, 1.],  # 6 front, right, down
                          [-half_length, half_width,-half_height, 1.],  # 7 back,  right, down
                          [-half_length, half_width, half_height, 1.]]) # 8 back,  left,  down


def wait_intil_ros_ready(ros, timeout = 10):
    """ pause until ros is ready or timeout reached """
    rospy.loginfo("waiting for ros...")
    while ros.latest_time is None or ros.quad_pose_gt is None or ros.pose_ego_w is None:
        if ros.latest_time is None:
            print("latest_time")
        if ros.pose_ego_w is None:
            print("pose_ego_w")
        if ros.pose_ego_w is None:
            print("pose_ego_w")
            time.sleep(0.1)
        continue
    rospy.loginfo("done!")


class camera:
    def __init__(self, ros):
         # camera intrinsic matrix K and pose relative to the quad (fixed)
        self.K , self.tf_cam_ego = get_ros_camera_info()

    def pnt3d_to_pix(self, pnt_c):
        """
        input: assumes pnt in camera frame
        output: [row, col] i.e. the projection of xyz onto camera plane
        """
        rc = self.K @ np.reshape(pnt_c[0:3], 3, 1)
        rc = np.array([rc[1], rc[0]]) / rc[2]
        return rc


if __name__ == '__main__':
    np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
    print("Starting MSL-RAPTOR main [running python {}]".format(sys.version_info[0]))
    rospy.init_node('RAPTOR_MSL', anonymous=True)
    run_execution_loop()
    print("--------------- FINISHED ---------------")

