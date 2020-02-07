# IMPORTS
# system
import sys, time
import pdb
# math
import numpy as np
# ros
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from msl_raptor.msg import angled_bb
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf
# libs & utils
from utils_msl_raptor.ros_utils import *
from utils_msl_raptor.ukf_utils import *
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ros_interface:

    def __init__(self, b_use_gt_bb=False):
        
        self.VERBOSE = True

        # Paramters #############################
        self.latest_img_time = -1
        self.DETECT = 1
        self.TRACK = 2
        self.FAKED_BB = 3
        self.IGNORE = 4
        self.im_seg_mode = self.DETECT
        # self.latest_abbs = None  # angled bound box [row, height, width, height, angle (radians)]
        # self.latest_bb_method = self.DETECT
        self.im_process_output = []  # what is accessed by the main function after an image is processed
        self.clas_str_to_id = {'mslquad' : 80, 'person' : 0} # this should be loaded from .names

        self.ego_pose_rosmesg_buffer = ([], [])
        self.ego_pose_rosmesg_buffer_len = 50
        self.ego_pose_gt_rosmsg = None

        self.img_seg = None  # object for parsing images into angled bounding boxes
        self.b_use_gt_bb = b_use_gt_bb  # toggle for debugging using ground truth bounding boxes
        ####################################################################

        self.ns = rospy.get_param('~ns')  # robot namespace

        self.bridge = CvBridge()
        self.camera = None

        # DEBUGGGGGGGGG
        if b_use_gt_bb:
            self.ado_pose_gt_rosmsg = None
            rospy.Subscriber('/quad4' + '/mavros/vision_pose/pose', PoseStamped, self.ado_pose_gt_cb, queue_size=10)  # DEBUG ONLY - optitrack pose
        ##########################
    
    def get_first_image(self):
        return self.bridge.imgmsg_to_cv2(rospy.wait_for_message(self.ns + '/camera/image_raw',Image), desired_encoding="passthrough")

    def create_subs_and_pubs(self):
        # Subscribers / Listeners & Publishers #############################   
        rospy.Subscriber(self.ns + '/camera/image_raw', Image, self.image_cb, queue_size=1,buff_size=2**21)
        rospy.Subscriber(self.ns + '/mavros/local_position/pose', PoseStamped, self.ego_pose_ekf_cb, queue_size=10)  # internal ekf pose
        rospy.Subscriber(self.ns + '/mavros/vision_pose/pose', PoseStamped, self.ego_pose_gt_cb, queue_size=10)  # optitrack pose
        self.state_pub = rospy.Publisher(self.ns + '/msl_raptor_state', PoseStamped, queue_size=5)
        self.bb_data_pub = rospy.Publisher(self.ns + '/bb_data', angled_bb, queue_size=5)
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
        my_time = get_ros_time(msg)  # time in seconds

        if len(self.ego_pose_rosmesg_buffer[0]) < self.ego_pose_rosmesg_buffer_len:
            self.ego_pose_rosmesg_buffer[0].append(msg.pose)
            self.ego_pose_rosmesg_buffer[1].append(my_time)
        else:
            self.ego_pose_rosmesg_buffer[0][0:self.ego_pose_rosmesg_buffer_len] = self.ego_pose_rosmesg_buffer[0][1:self.ego_pose_rosmesg_buffer_len]
            self.ego_pose_rosmesg_buffer[1][0:self.ego_pose_rosmesg_buffer_len] = self.ego_pose_rosmesg_buffer[1][1:self.ego_pose_rosmesg_buffer_len]
            self.ego_pose_rosmesg_buffer[0][-1] = msg.pose
            self.ego_pose_rosmesg_buffer[1][-1] = my_time


    def image_cb(self, msg):
        """
        receive an image, process w/ NN, then set variables that the main function will access 
        note: set the time variable at the end (this signals the program when new data has arrived)
        """
        tic = time.time()
        my_time = get_ros_time(msg)   # timestamp in seconds of msg

        # if self.im_seg_mode == self.IGNORE:
        #     return

        if len(self.ego_pose_rosmesg_buffer[0]) == 0:
            return # this happens if we are just starting

        self.tf_w_ego = pose_to_tf(find_closest_by_time(my_time, self.ego_pose_rosmesg_buffer[1], self.ego_pose_rosmesg_buffer[0])[0])

        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
        
        # undistort the fisheye effect in the image
        if self.camera is not None:
            image = cv2.undistort(image, self.camera.K, self.camera.dist_coefs, None, self.camera.new_camera_matrix)
        
        self.process_image(image)
        
            
        self.latest_img_time = my_time  # DO THIS LAST
        # self.img_seg_mode = self.IGNORE
        print("Image Callback time: {:.4f}".format(time.time() - tic))


    def process_image(self, image):
        self.latest_bb_method = self.im_seg_mode
        if not self.b_use_gt_bb:
            if self.im_seg_mode == self.DETECT:
                bbs_no_angle = self.img_seg.detect(image)  # returns a list of tuples: [(bb, class conf, object conf, class_id), ...]
                # num_detections = len(bbs_no_angle)
                if bbs_no_angle is False:
                    self.im_process_output = []
                    self.latest_img_time = -1
                    rospy.loginfo("Did not detect object")
                    return
                self.img_seg.reinit_tracker(bbs_no_angle, image) 
                output_tups = self.img_seg.track(image)  # returns a tuple of lists: ([abb1, abb2...], o_type1, o_type2..., [o_id1, o_id2...])
                self.im_process_output = [(bb_corners_to_angled_bb(abb.reshape(-1,2)), o_type, o_id, valid) for abb, o_type, o_id, valid in output_tups]

            elif self.im_seg_mode == self.TRACK:
                output_tups = self.img_seg.track(image)
                self.im_process_output = [(bb_corners_to_angled_bb(abb.reshape(-1,2)), o_type, o_id, valid) for abb, o_type, o_id, valid in output_tups]
            else:
                raise RuntimeError("Unknown image segmentation mode")
        else:
            self.im_seg_mode = self.FAKED_BB


            ros.publish_filter_state(obj_id, ukf.mu, ukf.itr_time, ukf.itr)  # send vector with time, iteration, state_est
    def publish_filter_state(self, obj_id, state_est, my_time, itr):
        """
        Broadcast the estimated state of the filter. 
        State assumed to be a Nx1 numpy array of floats
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time(my_time)
        pose_msg.header.frame_id = '{}'.format(obj_id)  # this is an int defining which object this is
        pose_msg.header.seq = np.uint32(itr)
        pose_msg.pose.position.y = state_est[1]
        pose_msg.pose.position.z = state_est[2]
        pose_msg.pose.orientation.w = state_est[6]
        pose_msg.pose.orientation.x = state_est[7]
        pose_msg.pose.orientation.y = state_est[8]
        pose_msg.pose.orientation.z = state_est[9]
        self.state_pub.publish(pose_msg)


    def publish_bb_msg(self, obj_id, bb, bb_seg_mode, bb_ts):
        """
        publish custom message type for angled bounding box
        """
        bb_msg = angled_bb()
        bb_msg.header.stamp = rospy.Time.from_sec(bb_ts)
        bb_msg.header.frame_id = '{}'.format(obj_id)  # this is an int defining which object this is
        bb_msg.x = bb[0]
        bb_msg.y = bb[1]
        bb_msg.width = bb[2]
        bb_msg.height = bb[3]
        bb_msg.angle = bb[4]
        bb_msg.im_seg_mode = bb_seg_mode
        self.bb_data_pub.publish(bb_msg)
