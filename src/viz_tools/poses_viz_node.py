#!/usr/bin/env python3
# IMPORTS
# system
import sys, time
from copy import copy
import pdb
# math
import numpy as np
# ros
import rospy
from msl_raptor.msg import TrackedObjects, TrackedObject
from geometry_msgs.msg import PoseArray
import tf

class poses_viz_node:

    def __init__(self):
        rospy.init_node('poses_viz_node', anonymous=True)

        self.ns = rospy.get_param('~ns')  # robot namespace
       
        self.tracked_objects_sub = rospy.Subscriber(self.ns + '/msl_raptor_state', TrackedObjects, self.tracked_objects_cb, queue_size=5)
        self.pose_array_pub = rospy.Publisher(self.ns + '/tracked_objects_poses', PoseArray, queue_size=5)
        

    def tracked_objects_cb(self, tracked_objects_msg):
        if len(tracked_objects_msg) == 0:
            return

        pose_arr = PoseArray()
        pose_arr.header = tracked_objects_msg.tracked_objects[0].pose.header

        for tracked_obj in tracked_objects_msg.tracked_objects:
            pose_arr.poses.append(tracked_obj.pose.pose)

        self.pose_array_pub.publish(pose_arr)


if __name__ == '__main__':
    try:
        program = poses_viz_node()
        rospy.spin()
    except:
        import traceback
        traceback.print_exc()

