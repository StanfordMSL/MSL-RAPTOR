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
import std_msgs
from geometry_msgs.msg import PoseStamped, Twist, Pose
from std_msgs.msg import Header
# other
import yaml


class gt_pose_broadcaster:
    def __init__(self):
        rospy.init_node('gt_pose_broadcaster', anonymous=True)
        rate = rospy.Rate(60)
        ns = rospy.get_param('~ns')
        gt_poses_to_broadcast_yaml = rospy.get_param('~gt_poses_to_broadcast_yaml')

        # create tf publisher & timer
        self.wfb = tf2_ros.TransformBroadcaster()
        self.gt_obs = []
        with open(gt_poses_to_broadcast_yaml, 'r') as stream:
            try:
                gt_poses = list(yaml.load_all(stream))
                print("broadcasting gt pose for {} objects".format(len(gt_poses)))
                for gt_object_dict in gt_poses:
                    obj_ns = gt_object_dict["ns"]
                    topic = "/{}/mavros/vision_pose/pose".format(obj_ns)
                    p = Pose()
                    p.position.x = gt_object_dict["x"]
                    p.position.y = gt_object_dict["y"]
                    p.position.z = gt_object_dict["z"]
                    p.orientation.x = gt_object_dict["qx"]
                    p.orientation.y = gt_object_dict["qy"]
                    p.orientation.z = gt_object_dict["qz"]
                    p.orientation.w = gt_object_dict["qw"]
                    frame = self.frame_from_pose(p, child_frame_id=obj_ns)
                    h = std_msgs.msg.Header()
                    h.stamp = rospy.Time.now()
                    # print(h.stamp)
                    h.frame_id = "world"
                    ps = PoseStamped()
                    ps.header = h
                    ps.pose = p
                    publisher = rospy.Publisher(topic, PoseStamped, queue_size=1)
                    self.gt_obs.append((p, ps, frame, publisher))
            except yaml.YAMLError as exc:
                print(exc)

        # pdb.set_trace()
        
        while not rospy.is_shutdown():
            self.publish_gts()
            rate.sleep()

    def publish_gts(self):
        for p, ps, frame, publisher in self.gt_obs:
            publisher.publish(ps)
            # print(ps)


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


