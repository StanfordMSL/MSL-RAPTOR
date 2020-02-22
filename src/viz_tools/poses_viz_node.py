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
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROSImage
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
# Utils
sys.path.append('/root/msl_raptor_ws/src/msl_raptor/src/utils_msl_raptor')
from ros_utils import find_closest_by_time
from ssp_utils import *

class poses_viz_node:

    def __init__(self):
        rospy.init_node('poses_viz_node', anonymous=True)

        self.ns = rospy.get_param('~ns')  # robot namespace
        self.b_save_3d_bb_img = rospy.get_param('~b_save_3d_bb_img')  # robot namespace
       
        self.tracked_objects_sub = rospy.Subscriber(self.ns + '/msl_raptor_state', TrackedObjects, self.tracked_objects_cb, queue_size=5)
        self.pose_array_pub = rospy.Publisher(self.ns + '/tracked_objects_poses', PoseArray, queue_size=5)
        self.img_index = 0
        
        # for displaying the 3d bb projected onto an image
        self.b_overlay = rospy.get_param('~b_overlay')  # robot namespace
        if self.b_overlay:
            self.bridge = CvBridge()
            self.img_buffer = ([], [])
            self.img_rosmesg_buffer_len = 200

            camera_info = rospy.wait_for_message(self.ns + '/camera/camera_info', CameraInfo, 30)
            self.K = np.reshape(camera_info.K, (3, 3))

            rospy.Subscriber(self.ns + '/camera/image_raw', ROSImage, self.image_cb, queue_size=1, buff_size=2**21)
            self.img_overlay_pub = rospy.Publisher(self.ns + '/image_3dbb_overlay', ROSImage, queue_size=5)
            
            if len(camera_info.D) == 5:
                self.dist_coefs = np.reshape(camera_info.D, (5,))
                self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (camera_info.width, camera_info.height), 0, (camera_info.width, camera_info.height))
            else:
                self.dist_coefs = None
                self.new_camera_matrix = self.K
            self.all_white_image = 255 * np.ones((camera_info.height, camera_info.width, 3), np.uint8)
            self.img_overlay_pub.publish(self.bridge.cv2_to_imgmsg(self.all_white_image, "bgr8"))


    def image_cb(self, msg):
        """
        Maintains a buffer of images & times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """
        my_time = msg.header.stamp.to_sec()  # time in seconds

        if len(self.img_buffer[0]) < self.img_rosmesg_buffer_len:
            self.img_buffer[0].append(msg)
            self.img_buffer[1].append(my_time)
        else:
            self.img_buffer[0][0:self.img_rosmesg_buffer_len] = self.img_buffer[0][1:self.img_rosmesg_buffer_len]
            self.img_buffer[1][0:self.img_rosmesg_buffer_len] = self.img_buffer[1][1:self.img_rosmesg_buffer_len]
            self.img_buffer[0][-1] = msg
            self.img_buffer[1][-1] = my_time


    def tracked_objects_cb(self, tracked_objects_msg):
        if len(tracked_objects_msg.tracked_objects) == 0:
            return

        pose_arr = PoseArray()
        pose_arr.header = tracked_objects_msg.tracked_objects[0].pose.header

        if self.b_overlay:
            my_time = pose_arr.header.stamp.to_sec()
            im_msg, pos_in_queue = find_closest_by_time(my_time, self.img_buffer[1], self.img_buffer[0])
            print("queue pos: {}".format(pos_in_queue))
            
            image = self.bridge.imgmsg_to_cv2(im_msg, desired_encoding="bgr8")
            image = cv2.undistort(image, self.K, self.dist_coefs, None, self.new_camera_matrix)

        for tracked_obj in tracked_objects_msg.tracked_objects:
            pose_arr.poses.append(tracked_obj.pose.pose)
            if self.b_overlay and not len(tracked_obj.projected_3d_bb) == 0:
                proj_3d_bb = np.reshape(tracked_obj.projected_3d_bb, (int(len(tracked_obj.projected_3d_bb)/2), 2) )
                image = self.draw_2d_proj_of_3D_bounding_box(image, corners2D_pr=proj_3d_bb)
            

        self.pose_array_pub.publish(pose_arr)
        if self.b_overlay:
            self.img_overlay_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            fn_str = "overlaid_raptor_img_{:d}".format(self.img_index)
            self.img_index += 1
            if self.b_save_3d_bb_img:
                cv2.imwrite("/mounted_folder/raptor_processed_bags/output_imgs/" + fn_str + ".jpg", image)


    def draw_2d_proj_of_3D_bounding_box(self, open_cv_image, corners2D_pr, corners2D_gt=None):
        """
        corners2D_gt/corners2D_pr is a 8x2 numpy array
        corner order (from ado frame):
          # 1 front, left,  up
          # 2 front, right, up
          # 3 back,  right, up
          # 4 back,  left,  up
          # 5 front, left,  down
          # 6 front, right, down
          # 7 back,  right, down
          # 8 back,  left,  down
        """

        dot_radius = 2
        color_list = [(255, 0, 0),     # 0 blue: center
                      (0, 255, 0),     # 1 green: front lower right
                      (0, 0, 255),     # 2 red: front upper right
                      (255, 255, 0),   # 3 cyan: front lower left
                      (255, 0, 255),   # 4 magenta: front upper left
                      (0, 255, 255),   # 5 yellow: back lower right
                      (0, 0, 0),       # 6 black: back upper right
                      (255, 255, 255), # 7 white: back lower left
                      (125, 125, 125)] # 8 grey: back upper left
        # if corners2D_gt is not None:
        #     for i, pnt in enumerate(corners2D_gt):
        #         pnt = np.round(pnt).astype(int)
        #         open_cv_image = cv2.circle(open_cv_image, (pnt[0], pnt[1]), dot_radius, color_list[i], -1)  # -1 means filled in, else edge thickness
        # if corners2D_pr is not None:
        #     for i, pnt in enumerate(corners2D_pr):
        #         pnt = np.round(pnt).astype(int)
        #         open_cv_image = cv2.circle(open_cv_image, (pnt[0], pnt[1]), dot_radius, color_list[i], -1)  # -1 means filled in, else edge thickness


        inds_to_connect = [[0, 3], [3, 2], [2, 1], [1, 0], # front face edges
                           [7, 4], [4, 5], [5, 6], [6, 7], # back face edges
                           [3, 4], [2, 5], [0, 7], [1, 6]] # side edges
        
        color_gt = (255,0,0)  # blue
        color_pr = (0,0,255)  # red
        linewidth = 1
        corners2D_pr = np.round(corners2D_pr).astype(int)
        for inds in inds_to_connect:

            if corners2D_gt is not None:
                open_cv_image = cv2.line(open_cv_image, (corners2D_gt[inds[0],0], corners2D_gt[inds[0],1]), (corners2D_gt[inds[1],0], corners2D_gt[inds[1], 1]), color_gt, linewidth)

            open_cv_image = cv2.line(open_cv_image, (corners2D_pr[inds[0],0], corners2D_pr[inds[0],1]), (corners2D_pr[inds[1],0], corners2D_pr[inds[1], 1]), color_pr, linewidth)

        return open_cv_image


if __name__ == '__main__':
    try:
        program = poses_viz_node()
        rospy.spin()
    except:
        import traceback
        traceback.print_exc()

