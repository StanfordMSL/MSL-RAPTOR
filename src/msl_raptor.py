#!/usr/bin/env python

# IMPORTS
# system
import os, sys, argparse, time
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


def run_execution_loop():
    rate = rospy.Rate(100) # max filter rate
    b_target_in_view = True
    last_image_time = -1
    ros = ROS()  # create a ros interface object
    my_ros_based_pause(2)  # pause to allow ros to get initial messages
    ukf = UKF()  # create ukf object
    camera = init_objects(ros, ukf)

    rospy.logwarn('need actual 3d bounding box info')
    bb_3d = np.zeros((8, 3))

    state_est = np.zeros((13, ))

    loop_count = 0
    while not rospy.is_shutdown():
        loop_time = ros.latest_time
        if loop_time <= last_image_time:
            # this means we dont have new data yet
            continue
        # store data locally (so it doesnt get overwritten in ROS object)

        abb = ros.latest_bb  # angled bounding box
        ego_pose = ros.latest_ego_pose  # stored as a ros PoseStamped
        bb_aqq_method = ros.latest_bb_method  # 1 for detect network, -1 for tracking network

        rospy.loginfo("Recieved new image at time {:.4f}".format(ros.latest_time))

        # update ukf
        # print(loop_time)
        # print(loop_count)
        # print(state_est)
        # print(abb)
        # print(ego_pose)
        # print(bb_aqq_method)
        ukf.step_ukf(abb, bb_3d, pose_to_tf(ego_pose))
        ros.publish_filter_state(np.concatenate(([loop_time], [loop_count], state_est)))  # send vector with time, iteration, state_est
        # [optional] update plots
        rate.sleep()
        loop_count += 1


def init_objects(ros, ukf):
    # create camera object
    camera = {}
    camera['K'] = ros.K  # camera intrinsic matrix is recieved via ros
    camera['tf_cam_ego'] = np.eye(4)
    # camera['tf_cam_ego'][0:3, 0:3] = 
    rospy.logwarn('need camera orientation relative to optitrack frame. Load from yaml??')
    # https://github.com/StanfordMSL/uav_game/blob/tro_experiments/ec_quad_sim/ec_quad_sim/param/quad3_trans.yaml
    camera['tf_cam_ego'][0:3, 3] = [0.05, 0.0, 0.07]
    camera['tf_cam_ego'][0:3, 0:3] = np.array([[ 0.0,  0.0,  1.0], 
                                               [-1.0,  0.0,  0.0], 
                                               [ 0.0, -1.0,  0.0]])

    # init ukf state
    rospy.logwarn('using ground truth to initialize filter!')
    ukf.mu = pose_to_state_vec(ros.quad_pose_gt)
    return camera



def my_ros_based_pause(pause_time = 1):
    """ pause for given time """
    start_time = rospy.Time.now().to_sec()
    time = rospy.Time.now().to_sec()
    while (time - start_time < pause_time):
        time = rospy.Time.now().to_sec()


if __name__ == '__main__':
    print("Starting MSL-RAPTOR main")
    rospy.init_node('RAPTOR_MSL', anonymous=True)
    run_execution_loop()
    print("--------------- FINISHED ---------------")

