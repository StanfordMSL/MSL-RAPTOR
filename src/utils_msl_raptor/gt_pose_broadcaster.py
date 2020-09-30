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
import tf
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist, Pose
# custom modules
from ros_interface import ros_interface as ROS
from ukf import UKF
import yaml


class gt_pose_broadcaster:
    def __init__(self):
        rospy.init_node('gt_pose_broadcaster', anonymous=True)

        ns = rospy.get_param('~ns')
        gt_poses_to_broadcast_yaml = rospy.get_param('~gt_poses_to_broadcast_yaml')

        # create tf publisher & timer
        self.wfb = tf2_ros.TransformBroadcaster()
        gt_obs = []
        with open(gt_poses_to_broadcast_yaml, 'r') as stream:
            try:
                gt_poses = list(yaml.load_all(stream))
                for gt_object_dict in gt_poses:
                    obj_ns = gt_object_dict.ns
                    topic = "{}/mavros/vision_pose/pose".format(obj_ns)
                    p = Pose()
                    p.position.x = gt_object_dict.x
                    p.position.y = gt_object_dict.y
                    p.position.z = gt_object_dict.z
                    p.orientation.x = gt_object_dict.qx
                    p.orientation.y = gt_object_dict.qy
                    p.orientation.z = gt_object_dict.qz
                    p.orientation.w = gt_object_dict.qw
                    frame = self.frame_from_pose(p, child_frame_id=obj_ns)
                    pdb.set_trace()
        
        while not rospy.is_shutdown():
            self.publish_gt()
            rate.sleep()

    #     self.run()

    # def run(self):
    #     rate = rospy.Rate(100)
    #     while not rospy.is_shutdown():
    #         self.publish_custom_frames()
    #         rate.sleep()

    # def publish_custom_frames(self, event=None):
    #     if self.odom_frame:
    #         # publish frame connecting world to the odom frame
    #         self.wfb.sendTransform(self.odom_frame)

    def publish_gt(self, msg):
		
		new_tf = tf2_ros.TransformStamped()
		new_tf.header.frame_id = 'world'
		for ind, botname in enumerate(msg.name):
			if botname.find("ouijabot") < 0:
				continue
			botnum = re.sub("[^0-9]", "", botname)

			new_tf.child_frame_id  = 'ouijabot' + botnum + '/ground_truth'
			new_tf.header.stamp  = rospy.Time.now()

			new_tf.transform.translation.x = msg.pose[ind].position.x
			new_tf.transform.translation.y = msg.pose[ind].position.y
			new_tf.transform.translation.z = msg.pose[ind].position.z
			new_tf.transform.rotation.x    = msg.pose[ind].orientation.x
			new_tf.transform.rotation.y    = msg.pose[ind].orientation.y
			new_tf.transform.rotation.z    = msg.pose[ind].orientation.z
			new_tf.transform.rotation.w    = msg.pose[ind].orientation.w

			self.wfb.sendTransform(new_tf)

    def frame_from_pose(self, pose, child_frame_id, parent_frame_id='world'):
        frame = tf2_ros.TransformStamped()
        frame.header.frame_id = parent_frame_id
        frame.child_frame_id  = child_frame_id
        frame.header.stamp  = rospy.Time.now()

        frame.transform.translation.x = pose.position.x
        frame.transform.translation.y = pose.position.y
        frame.transform.translation.z = pose.position.z
        frame.transform.rotation.x    = pose.orientation.x
        frame.transform.rotation.y    = pose.orientation.y
        frame.transform.rotation.z    = pose.orientation.z
        frame.transform.rotation.w    = pose.orientation.w
        return frame

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


