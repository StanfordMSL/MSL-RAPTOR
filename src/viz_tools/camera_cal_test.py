#!/usr/bin/env python3
# IMPORTS
# system
import sys, time
from copy import copy
import pdb
# math
import numpy as np
from bisect import bisect_left
# ros
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf
# libs & utils
# from utils_msl_raptor.ros_utils import *
# from utils_msl_raptor.ukf_utils import *
import cv2
from cv_bridge import CvBridge, CvBridgeError

class camera_cal_test_node:
    """
    This rosnode has two modes. In mode 1 it publishes a white background with the bounding boxes drawn (green when tracking, red when detecting). 
    This is faster and less likely to slow down the network. Mode 2 publishes the actual image. This is good for debugging, but is slower.
    If rosparam b_overlay is false (default), it will be mode 1, else mode 2.
    """

    def __init__(self):
        rospy.init_node('camera_cal_test_node', anonymous=True)
        self.itr = 0
        self.bridge = CvBridge()

        self.ns = rospy.get_param('~ns')  # robot namespace
        self.overlaid_img = None
        
        camera_info = rospy.wait_for_message(self.ns + '/camera/camera_info', CameraInfo, 30)
        self.K = np.reshape(camera_info.K, (3, 3))
        
        rospy.Subscriber(self.ns + '/camera/image_raw', Image, self.image_cb, queue_size=1, buff_size=2**21)
        self.img_overlay_pub = rospy.Publisher(self.ns + '/image_bb_overlay', Image, queue_size=5)

        if len(camera_info.D) == 5:
            self.dist_coefs = np.reshape(camera_info.D, (5,))
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (camera_info.width, camera_info.height), 0, (camera_info.width, camera_info.height))
        else:
            self.dist_coefs = None
            self.new_camera_matrix = self.K

        self.img_buffer = ([], [])
        self.img_rosmesg_buffer_len = 500

        self.ego_pose_rosmesg_buffer = ([], [])
        self.ego_pose_rosmesg_buffer_gt = ([], [])
        self.ego_pose_rosmesg_buffer_len = 50
        self.ego_pose_gt_rosmsg = None

        self.ado_pose_rosmesg_buffer = ([], [])
        self.ado_pose_rosmesg_buffer_gt = ([], [])
        self.ado_pose_rosmesg_buffer_len = 50
        self.ado_pose_gt_rosmsg = None
        
        rospy.Subscriber('/vrpn_client_node/' + self.ns + 'pose', PoseStamped, self.ego_pose_gt_cb, queue_size=10)  # optitrack pose

        test_ado_object_name = 'bowl_green_msl'
        test_ado_object_topic = '/vrpn_client_node/' + test_ado_object_name + 'pose'
        rospy.Subscriber(test_ado_object_topic, PoseStamped, self.ado_pose_gt_cb, queue_size=10)
        # self.all_white_image = 255 * np.ones((camera_info.height, camera_info.width, 3), np.uint8)
        # self.img_overlay_pub.publish(self.bridge.cv2_to_imgmsg(self.all_white_image, "bgr8"))

    
    def ado_pose_gt_cb(self, msg):
        """
        Maintains a buffer of poses and times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """
        my_time = get_ros_time(msg)  # time in seconds

        if len(self.ado_pose_rosmesg_buffer_gt[0]) < self.ado_pose_rosmesg_buffer_len:
            self.ado_pose_rosmesg_buffer_gt[0].append(msg.pose)
            self.ado_pose_rosmesg_buffer_gt[1].append(my_time)
        else:
            self.ado_pose_rosmesg_buffer_gt[0][0:self.ado_pose_rosmesg_buffer_len] = self.ado_pose_rosmesg_buffer_gt[0][1:self.ado_pose_rosmesg_buffer_len]
            self.ado_pose_rosmesg_buffer_gt[1][0:self.ado_pose_rosmesg_buffer_len] = self.ado_pose_rosmesg_buffer_gt[1][1:self.ado_pose_rosmesg_buffer_len]
            self.ado_pose_rosmesg_buffer_gt[0][-1] = msg.pose
            self.ado_pose_rosmesg_buffer_gt[1][-1] = my_time

    def ego_pose_gt_cb(self, msg):
        # self.ego_pose_gt_rosmsg = msg.pose
        """
        Maintains a buffer of poses and times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """
        my_time = get_ros_time(msg)  # time in seconds

        if len(self.ego_pose_rosmesg_buffer_gt[0]) < self.ego_pose_rosmesg_buffer_len:
            self.ego_pose_rosmesg_buffer_gt[0].append(msg.pose)
            self.ego_pose_rosmesg_buffer_gt[1].append(my_time)
        else:
            self.ego_pose_rosmesg_buffer_gt[0][0:self.ego_pose_rosmesg_buffer_len] = self.ego_pose_rosmesg_buffer_gt[0][1:self.ego_pose_rosmesg_buffer_len]
            self.ego_pose_rosmesg_buffer_gt[1][0:self.ego_pose_rosmesg_buffer_len] = self.ego_pose_rosmesg_buffer_gt[1][1:self.ego_pose_rosmesg_buffer_len]
            self.ego_pose_rosmesg_buffer_gt[0][-1] = msg.pose
            self.ego_pose_rosmesg_buffer_gt[1][-1] = my_time


    def image_cb(self, msg):
        """
        Maintains a buffer of images & times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """
        # my_time = rospy.Time.now().to_sec()  # time in seconds
        my_time = msg.header.stamp.to_sec()  # time in seconds

        if len(self.img_buffer[0]) < self.img_rosmesg_buffer_len:
            self.img_buffer[0].append(msg)
            self.img_buffer[1].append(my_time)
        else:
            self.img_buffer[0][0:self.img_rosmesg_buffer_len] = self.img_buffer[0][1:self.img_rosmesg_buffer_len]
            self.img_buffer[1][0:self.img_rosmesg_buffer_len] = self.img_buffer[1][1:self.img_rosmesg_buffer_len]
            self.img_buffer[0][-1] = msg
            self.img_buffer[1][-1] = my_time


    def find_closest_by_time_ros2(self, time_to_match, time_list, message_list=None):
        """
        Assumes lists are sorted earlier to later. Returns closest item in list by time. If two numbers are equally close, return the smallest number.
        Adapted from https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
        """
        if not message_list:
            message_list = time_list
        pos = bisect_left(time_list, time_to_match)
        if pos == 0:
            t_delta = time_list[0]- time_to_match
            if t_delta > 0.0001:
                print("warning: \"bottoming out\" in buffer (match time = {:.4f}, closest queue time = {:.4f}, delta = {:.4f})".format(time_to_match, time_list[0], t_delta))
            return message_list[0], 0
        if pos == len(time_list):
            return message_list[-1], len(message_list) - 1
        before = time_list[pos - 1]
        after = time_list[pos]
        if after - time_to_match < time_to_match - before:
            return message_list[pos], pos
        else:
            return message_list[pos - 1], pos - 1


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
        num_abbs = len(bb_list_msg.boxes)
        if num_abbs == 0 or (self.b_overlay and (not self.img_buffer or len(self.img_buffer[0]) == 0)):
            return
        elif self.b_overlay:
            my_time = bb_list_msg.header.stamp.to_sec()
            im_msg, pos_in_queue = self.find_closest_by_time_ros2(my_time, self.img_buffer[1], self.img_buffer[0])
            print("queue pos: {}".format(pos_in_queue))
            image = self.bridge.imgmsg_to_cv2(im_msg, desired_encoding="bgr8")
            image = cv2.undistort(image, self.K, self.dist_coefs, None, self.new_camera_matrix)
        else:
            image = copy(self.all_white_image)

        for bb_msg in bb_list_msg.boxes:
            
            im_seg_mode = bb_msg.im_seg_mode
            if im_seg_mode == self.DETECT:
                box_color = (0,0,255)  # RED
            elif im_seg_mode == self.TRACK:
                box_color = (0,255,0)  # GREEN
            elif im_seg_mode == self.FAKED_BB:
                box_color = (255,0,0)  # BLUE
                if self.itr % 50 == 0:
                    print("simulating bounding box - DEBUGGING ONLY")
            else:
                print("not detecting nor tracking! (seg mode: {})".format(im_seg_mode))
                box_color = (255,0,0)
            # bb_data = msg.data[0:-2]

            box = np.int0(cv2.boxPoints( ( (bb_msg.x, bb_msg.y), (bb_msg.width, bb_msg.height), -np.degrees(bb_msg.angle))) )
            cv2.drawContours(image, [box], 0, box_color, 2)
        self.img_overlay_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
        if False:
            cv2.imwrite('/mounted_folder/front_end_imgs/front_end_{}.png'.format(self.itr), image)
        self.itr += 1
        # pdb.set_trace()


    def run(self):
        rate = rospy.Rate(15)

        while not rospy.is_shutdown():
            if len(self.img_buffer[0]) > 0:
                img_msg = self.img_buffer[0][-1]
                img_time = self.img_buffer[1][-1]
                ego_pose_msg, pos_in_queue_ego = self.find_closest_by_time_ros2(img_time, self.ego_pose_rosmesg_buffer_gt[1], self.ego_pose_rosmesg_buffer_gt[0])
                ado_pose_msg, pos_in_queue_ado = self.find_closest_by_time_ros2(img_time, self.ado_pose_rosmesg_buffer_gt[1], self.ado_pose_rosmesg_buffer_gt[0])

                # generate corners in ado frame
                corners = []
                for x in [-0.5, 0.5]:
                    for y in [-0.5, 0.5]:
                        for z in [-0.5, 0.5]:
                            corners.append(np.array([x, y, z, 1]))
                corners = np.stack(corners, axis=-1)

                corners_cam_frame = T_c_a @ corners
                bb_pix = self.new_camera_matrix @ corners_cam_frame
                

                image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
                image = cv2.undistort(image, self.K, self.dist_coefs, None, self.new_camera_matrix)


                self.img_overlay_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            rate.sleep()


if __name__ == '__main__':
    try:
        program = camera_cal_test_node()
        program.run()
    except:
        import traceback
        traceback.print_exc()

