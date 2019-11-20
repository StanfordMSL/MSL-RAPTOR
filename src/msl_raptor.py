#!/usr/bin/env python

"""
The MIT License (MIT)
Copyright (c) 2019 MSL
"""

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
# libs
from ros_interface import ros_interface as ROS

def run_execution_loop():
    rate = rospy.Rate(100) # max filter rate
    b_target_in_view = True
    last_image_time = -1
    ros = ROS()  # create a ros interface object
    state_est = np.zeros((13, 1))

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
        ros.publish_filter_state(np.concatenate((loop_time, loop_count, ros.state_est)))  # send vector with time, iteration, state_est
        # [optional] update plots
        rate.sleep()
        loop_count += 1    


if __name__ == '__main__':
    print("Starting MSL-RAPTOR main")
    rospy.init_node('RAPTOR_MSL', anonymous=True)
    run_execution_loop()
    print("--------------- FINISHED ---------------")


            