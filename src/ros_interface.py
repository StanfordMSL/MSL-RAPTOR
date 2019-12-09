# IMPORTS
# system
import sys, time
import pdb
# math
import numpy as np
# ros
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf
# libs & utils
from utils_msl_raptor.ros_utils import *
from utils_msl_raptor.ukf_utils import *
from cv_bridge import CvBridge, CvBridgeError

class ros_interface:

    def __init__(self, b_use_gt_bb=False):
        
        self.VERBOSE = True

        # Paramters #############################
        self.start_time = None
        self.latest_img_time = -1
        self.DETECT = 1
        self.TRACK = 2
        self.REINIT = 3
        self.im_seg_mode = self.DETECT
        self.latest_abb = None  # angled bound box [row, height, width, height, angle (radians)]
        self.latest_bb_method = 1  # 1 for detect network, -1 for tracking network

        self.ego_pose_rosmesg_buffer = ([], [])
        self.ego_pose_rosmesg_buffer_len = 50
        self.ego_pose_gt_rosmsg = None

        self.img_seg = None  # object for parsing images into angled bounding boxes
        self.b_use_gt_bb = b_use_gt_bb  # toggle for debugging using ground truth bounding boxes
        ####################################################################

        self.ns = rospy.get_param('~ns')  # robot namespace

        self.bridge = CvBridge()
        # DEBUGGGGGGGGG
        if b_use_gt_bb or not b_use_gt_bb:
            rospy.logwarn("!!!ALWAYS INITIALIZING WITH GT POSE!!!!!!")
            self.ado_pose_gt_rosmsg = None
            rospy.Subscriber('/quad4' + '/mavros/vision_pose/pose', PoseStamped, self.ado_pose_gt_cb, queue_size=10)  # DEBUG ONLY - optitrack pose

        ##########################
    
    def get_first_image(self):
        return self.bridge.imgmsg_to_cv2(rospy.wait_for_message(self.ns + '/camera/image_raw',Image), desired_encoding="passthrough")

    def create_subs_and_pubs(self):
        # Subscribers / Listeners & Publishers #############################   
        rospy.Subscriber(self.ns + '/camera/image_raw', Image, self.image_cb)
        rospy.Subscriber(self.ns + '/mavros/local_position/pose', PoseStamped, self.ego_pose_ekf_cb, queue_size=10)  # internal ekf pose
        rospy.Subscriber(self.ns + '/mavros/vision_pose/pose', PoseStamped, self.ego_pose_gt_cb, queue_size=10)  # optitrack pose
        self.state_pub = rospy.Publisher(self.ns + '/msl_raptor_state', PoseStamped, queue_size=5)
        ####################################################################

    def ado_pose_gt_cb(self, msg):
        self.ado_pose_gt_rosmsg = msg.pose


    def ego_pose_gt_cb(self, msg):
        self.ego_pose_gt_rosmsg = msg.pose


    def ego_pose_ekf_cb(self, msg):
        """
        Maintains a buffer of poses and times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """
        if self.start_time is None:
            return

        time = get_ros_time(self.start_time)  # time in seconds

        if len(self.ego_pose_rosmesg_buffer[0]) < self.ego_pose_rosmesg_buffer_len:
            self.ego_pose_rosmesg_buffer[0].append(msg.pose)
            self.ego_pose_rosmesg_buffer[1].append(time)
        else:
            self.ego_pose_rosmesg_buffer[0][0:self.ego_pose_rosmesg_buffer_len] = self.ego_pose_rosmesg_buffer[0][1:self.ego_pose_rosmesg_buffer_len]
            self.ego_pose_rosmesg_buffer[1][0:self.ego_pose_rosmesg_buffer_len] = self.ego_pose_rosmesg_buffer[1][1:self.ego_pose_rosmesg_buffer_len]
            self.ego_pose_rosmesg_buffer[0][-1] = msg.pose
            self.ego_pose_rosmesg_buffer[1][-1] = time


    def image_cb(self, msg):
        """
        receive an image, process w/ NN, then set variables that the main function will access 
        note: set the time variable at the end (this signals the program when new data has arrived)
        """
        if self.start_time is None:
            self.start_time = msg.header.stamp.to_sec()
            time = 0
        else:
            time = get_ros_time(self.start_time, msg)   # timestamp in seconds

        if len(self.ego_pose_rosmesg_buffer[0]) == 0:
            return # this happens if we are just starting

        self.tf_w_ego = pose_to_tf(find_closest_by_time(time, self.ego_pose_rosmesg_buffer[1], self.ego_pose_rosmesg_buffer[0])[0])

        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
        self.latest_bb_method = self.im_seg_mode
        if not self.b_use_gt_bb:
            if self.im_seg_mode == self.DETECT:
                bb_no_angle = self.img_seg.detect(image)
                if not bb_no_angle:
                    rospy.loginfo("Did not detect object")
                    return
                self.img_seg.reinit_tracker(bb_no_angle, image)
                self.latest_abb = self.img_seg.track(image)
                self.latest_abb = bb_corners_to_angled_bb(self.latest_abb.reshape(-1,2))
            elif self.im_seg_mode == self.TRACK:
                self.latest_abb = self.img_seg.track(image)
                self.latest_abb = bb_corners_to_angled_bb(self.latest_abb.reshape(-1,2))
            else:
                raise RuntimeError("Unknown image segmentation mode")
            
        self.latest_img_time = time  # DO THIS LAST


    def publish_filter_state(self, state_est, time, itr):
        """
        Broadcast the estimated state of the filter. 
        State assumed to be a Nx1 numpy array of floats
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time(time)
        pose_msg.header.frame_id = 'world'
        pose_msg.header.seq = np.uint32(itr)
        pose_msg.pose.position.x = state_est[0]
        pose_msg.pose.position.y = state_est[1]
        pose_msg.pose.position.z = state_est[2]
        pose_msg.pose.orientation.w = state_est[6]
        pose_msg.pose.orientation.x = state_est[7]
        pose_msg.pose.orientation.y = state_est[8]
        pose_msg.pose.orientation.z = state_est[9]
        self.state_pub.publish(pose_msg)
