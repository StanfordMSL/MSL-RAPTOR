#!/usr/bin/env python3

# IMPORTS
# system
import os, sys, argparse, time
import pdb
# math
import numpy as np
import cv2
# ros
import rospy
# custom modules
from ros_interface import ros_interface as ROS
from ukf import UKF
import yaml


class gt_pose_broadcaster:
    def __init__(self):
        rospy.init_node('gt_pose_broadcaster', anonymous=True)

        gt_poses_to_broadcast_yaml = rospy.get_param('~gt_poses_to_broadcast_yaml')
        
        with open(gt_poses_to_broadcast_yaml, 'r') as stream:
            try:
                gt_poses = list(yaml.load_all(stream))
                for gt_object_dict in gt_poses:
                    pbd.set_trace()
                    pass



if __name__ == '__main__':
    np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
    try:
        program = gt_pose_broadcaster()
    except:
        import traceback
        traceback.print_exc()
    print("--------------- FINISHED gt pose broadcaster---------------")

# class bb_viz_node:

#     def __init__(self):
#         rospy.init_node('bb_viz_node', anonymous=True)
#         self.itr = 0
#         self.DETECT = 1
#         self.TRACK = 2
#         self.FAKED_BB = 3
#         self.IGNORE = 4


#     def run(self):
#         rate = rospy.Rate(15)

#         while not rospy.is_shutdown():
#             rate.sleep()


