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
from msl_raptor.msg import AngledBbox, AngledBboxes, TrackedObjects, TrackedObject
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
from ukf_utils import verts_to_angled_bb
from viz_utils import *

class poses_viz_node:

    def __init__(self):
        rospy.init_node('poses_viz_node', anonymous=True)

        self.ns = rospy.get_param('~ns')  # robot namespace

        self.b_draw_2d_bbs_on_3dbb_proj = True
       
        self.tracked_objects_sub = rospy.Subscriber(self.ns + '/msl_raptor_state', TrackedObjects, self.tracked_objects_cb, queue_size=5)
        self.pose_array_pub = rospy.Publisher(self.ns + '/tracked_objects_poses', PoseArray, queue_size=5)
        self.bb_data_sub = rospy.Subscriber(self.ns + '/bb_data', AngledBboxes, self.bb_viz_cb, queue_size=5)
        self.bb_msg_buf = ([], [])
        self.bb_msg_buf_maxlen = 50
        
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
                if len(tracked_obj.connected_inds) > 0:
                    connected_inds = np.reshape(tracked_obj.connected_inds, (int(len(tracked_obj.connected_inds)/2), 2) )
                    image = draw_2d_proj_of_3D_bounding_box(image, corners2D_pr=proj_3d_bb, inds_to_connect=connected_inds)
                else:
                    image = draw_2d_proj_of_3D_bounding_box(image, corners2D_pr=proj_3d_bb)
                
                if self.b_draw_2d_bbs_on_3dbb_proj:
                    abb = verts_to_angled_bb(proj_3d_bb)
                    box = np.int0(cv2.boxPoints( ( (abb[0], abb[1]), (abb[2], abb[3]), -np.degrees(abb[4]))) )
                    cv2.drawContours(image, [box], 0, (255,0,0), 2)
                    
                    bb_msg, _ = find_closest_by_time(my_time, self.bb_msg_buf[1], self.bb_msg_buf[0])
                    box = np.int0(cv2.boxPoints( ( (bb_msg.x, bb_msg.y), (bb_msg.width, bb_msg.height), -np.degrees(bb_msg.angle))) )
                    cv2.drawContours(image, [box], 0, (0,255,0), 2)
            

        self.pose_array_pub.publish(pose_arr)
        if self.b_overlay:
            self.img_overlay_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))


    def bb_viz_cb(self, bb_list_msg):
        """
        custom list of angled bb message type:
            Header header  # ros timestamp etc
            float64 x  # x coordinate of center of bounding box
            float64 y  # y coordinate of center of bounding box
            float64 width  # width (long axis) of bounding box
            float64 height  # height (short axis) of bounding box
            float64 angle  # angle in radians
            uint8 im_seg_mode  # self.DETECT = 1  self.TRACK = 2  self.FAKED_BB = 3  self.IGNORE = 4
        """
        # my_time = rospy.Time.now().to_sec()  # time in seconds
        my_time = bb_list_msg.header.stamp.to_sec()  # time in seconds

        if len(self.bb_msg_buf[0]) < self.bb_msg_buf_maxlen:
            self.bb_msg_buf[0].append(bb_list_msg.boxes[0])
            self.bb_msg_buf[1].append(my_time)
        else:
            self.bb_msg_buf[0][0:self.bb_msg_buf_maxlen] = self.bb_msg_buf[0][1:self.bb_msg_buf_maxlen]
            self.bb_msg_buf[1][0:self.bb_msg_buf_maxlen] = self.bb_msg_buf[1][1:self.bb_msg_buf_maxlen]
            self.bb_msg_buf[0][-1] = bb_list_msg.boxes[0]
            self.bb_msg_buf[1][-1] = my_time


if __name__ == '__main__':
    try:
        program = poses_viz_node()
        rospy.spin()
    except:
        import traceback
        traceback.print_exc()

