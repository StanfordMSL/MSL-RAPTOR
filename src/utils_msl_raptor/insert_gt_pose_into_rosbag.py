#!/usr/bin/env python3
# IMPORTS
# system
import sys, os, time
from copy import copy
from collections import defaultdict
import yaml
import pdb
# math
import numpy as np
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
# ros
import rosbag
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
import std_msgs
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from msl_raptor.msg import AngledBbox, AngledBboxes, TrackedObjects, TrackedObject
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError

class insert_gt_pose_into_rosbag:
    def __init__(self, rb_path_and_name, ego_ns, b_ego_pose_est, gt_poses_to_broadcast_yaml):
        self.gt_obs = []
        self.read_gt_poses(gt_poses_to_broadcast_yaml)

        bag_out = rosbag.Bag(rb_path_and_name + "_with_gt.bag", 'w')
        bag_in = rosbag.Bag(rb_path_and_name + ".bag", 'r')
        for topic, msg, t in bag_in.read_messages():
            # print(t)
            # pdb.set_trace()
            # if topic.split('/')[1] == "vrpn_client_node":
            #     ns_tmp = topic.split('/')[2]
            #     bag_out.write("/" + ns_tmp + "/mavros/vision_pose/pose", msg, t)
            #     if b_ego_pose_est and ns_tmp == ego_ns:
            #         bag_out.write("/" + ns_tmp + "/mavros/local_position/pose", msg, t)
            # else:
            #     bag_out.write(topic, msg, t)
            if topic == "/{}/mavros/vision_pose/pose".format(ego_ns):
                # pdb.set_trace()
                for ps, gt_topic in self.gt_obs:
                    ps.header.stamp = t
                    bag_out.write(gt_topic, ps, t)
            bag_out.write(topic, msg, t)

        bag_in.close()
        # b_first_bag = False
        bag_out.close()


        # bag_out = rosbag.Bag(rb_path + rb_out_name, 'r')
        # for topic, msg, t in bag_out.read_messages():
        #     print("time: {}, topic: {}".format(t, topic))
        # bag_out.close()



    def read_gt_poses(self, gt_poses_to_broadcast_yaml):
        with open(gt_poses_to_broadcast_yaml, 'r') as stream:
            try:
                gt_poses = list(yaml.load_all(stream))
                print("broadcasting gt pose for {} objects".format(len(gt_poses)))
                for gt_object_dict in gt_poses:
                    obj_ns = gt_object_dict["ns"]
                    gt_topic = "/{}/mavros/vision_pose/pose".format(obj_ns)
                    p = Pose()
                    p.position.x = gt_object_dict["x"]
                    p.position.y = gt_object_dict["y"]
                    p.position.z = gt_object_dict["z"]
                    p.orientation.x = gt_object_dict["qx"]
                    p.orientation.y = gt_object_dict["qy"]
                    p.orientation.z = gt_object_dict["qz"]
                    p.orientation.w = gt_object_dict["qw"]
                    # frame = self.frame_from_pose(p, child_frame_id=obj_ns)
                    h = std_msgs.msg.Header()
                    # h.stamp = rospy.Time.now()
                    # print(h.stamp)
                    h.frame_id = "world"
                    ps = PoseStamped()
                    ps.header = h
                    ps.pose = p
                    self.gt_obs.append((ps, gt_topic))
            except yaml.YAMLError as exc:
                print(exc)



if __name__ == '__main__':
    try:
        np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
        # program = rosbags_to_logs(rb_name=my_rb_name, data_source=my_data_source, ego_yaml=my_ego_yaml, ado_yaml=my_ado_yaml, b_save_3dbb_imgs=my_b_save_3dbb_imgs)
        # rb_path_and_name_ = "/mounted_folder/raptor_processed_bags/msl_raptor_output_from_bag_rosbag_for_post_process_2020-09-29-16-32-38"
        rb_path_and_name_ = "/mounted_folder/rosbags_for_post_process/rosbag_for_post_process_2020-09-29-16-32-38"
        gt_poses_to_broadcast_yaml_ = "/root/msl_raptor_ws/src/msl_raptor/params/gt_poses_to_broadcast.yaml"
        ego_ns_ = "quad7"
        b_ego_pose_est_ = True

        rb_unified = insert_gt_pose_into_rosbag(rb_path_and_name=rb_path_and_name_, ego_ns=ego_ns_, b_ego_pose_est=b_ego_pose_est_, gt_poses_to_broadcast_yaml = gt_poses_to_broadcast_yaml_)
    except:
        import traceback
        traceback.print_exc()
    print("done with program!")