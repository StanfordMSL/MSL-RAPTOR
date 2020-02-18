# IMPORTS
# system
import sys, time
import pdb
# math
import numpy as np
# ros
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from msl_raptor.msg import AngledBbox,AngledBboxes,TrackedObjects,TrackedObject
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf
# libs & utils
from utils_msl_raptor.ros_utils import *
from utils_msl_raptor.ukf_utils import *
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ros_interface:

    def __init__(self, b_use_gt_bb=False,b_verbose=False,b_use_gt_init=False):
        
        self.verbose = b_verbose

        # Parameters #############################
        self.im_process_output = []  # what is accessed by the main function after an image is processed

        self.ego_pose_rosmesg_buffer = ([], [])
        self.ego_pose_rosmesg_buffer_len = 50
        self.ego_pose_gt_rosmsg = None

        self.im_seg = None  # object for parsing images into angled bounding boxes
        self.b_use_gt_bb = b_use_gt_bb  # toggle for debugging using ground truth bounding boxes
        self.latest_img_time = -1
        ####################################################################

        self.ns = rospy.get_param('~ns')  # robot namespace

        self.bridge = CvBridge()
        self.camera = None

        # DEBUGGGGGGGGG
        if b_use_gt_bb:
            self.ado_pose_gt_rosmsg = None
            rospy.Subscriber('/quad4' + '/mavros/vision_pose/pose', PoseStamped, self.ado_pose_gt_cb, queue_size=10)  # DEBUG ONLY - optitrack pose
        ##########################

        # Just used for initializing from groundtruth
        self.objects_names_per_class = None
        self.b_use_gt_init  = b_use_gt_init

    def get_first_image(self):
        return self.bridge.imgmsg_to_cv2(rospy.wait_for_message(self.ns + '/camera/image_raw',Image), desired_encoding="bgr8")

    def create_subs_and_pubs(self):
        # Subscribers / Listeners & Publishers #############################   
        rospy.Subscriber(self.ns + '/camera/image_raw', Image, self.image_cb, queue_size=1,buff_size=2**21)
        rospy.Subscriber(self.ns + '/mavros/local_position/pose', PoseStamped, self.ego_pose_ekf_cb, queue_size=10)  # internal ekf pose
        rospy.Subscriber(self.ns + '/mavros/vision_pose/pose', PoseStamped, self.ego_pose_gt_cb, queue_size=10)  # optitrack pose
        self.state_pub = rospy.Publisher(self.ns + '/msl_raptor_state', TrackedObjects, queue_size=5)
        self.bb_data_pub = rospy.Publisher(self.ns + '/bb_data', AngledBboxes, queue_size=5)

        if self.b_use_gt_init:
            # Create dict to store pose for each object
            self.latest_tracked_poses = {}
            for obj_name in self.objects_names_per_class.values():
                rospy.Subscriber(obj_name + '/mavros/vision_pose/pose', PoseStamped, self.tracked_objects_poses_cb, 'obj_name',queue_size=5)
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

        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
        
        # undistort the fisheye effect in the image
        if self.camera is not None:
            image = cv2.undistort(image, self.camera.K, self.camera.dist_coefs, None, self.camera.new_camera_matrix)
        
        self.latest_bb_method = self.im_seg.mode
        self.im_process_output = self.im_seg.process_image(image,my_time)

        self.latest_img_time = my_time  # DO THIS LAST
        # self.img_seg_mode = self.IGNORE
        if self.verbose:
            print("Image Callback time: {:.4f}".format(time.time() - tic))



    def publish_filter_state(self, obj_ids,ukf_dict):# state_est, my_time, itr):
        """
        Broadcast the estimated state of the filter. 
        State assumed to be a Nx1 numpy array of floats
        """
        tracked_objects = []
        for id in obj_ids:
            obj = TrackedObject()
            pose_msg = PoseStamped()
            state_est = ukf_dict[id].mu
            pose_msg.header.stamp = rospy.Time(ukf_dict[id].itr_time)
            pose_msg.header.frame_id = 'world'
            pose_msg.header.seq = np.uint32(ukf_dict[id].itr)
            pose_msg.pose.position.x = state_est[0]
            pose_msg.pose.position.y = state_est[1]
            pose_msg.pose.position.z = state_est[2]
            pose_msg.pose.orientation.w = state_est[6]
            pose_msg.pose.orientation.x = state_est[7]
            pose_msg.pose.orientation.y = state_est[8]
            pose_msg.pose.orientation.z = state_est[9]

            obj.pose = pose_msg
            obj.class_str = ukf_dict[id].class_str
            obj.id = id

            tracked_objects.append(obj)

        self.state_pub.publish(tracked_objects)


    def publish_bb_msg(self,processed_image, bb_seg_mode, bb_ts):
        """
        publish custom message type for angled bounding box
        """
        bb_list_msg = AngledBboxes()
        header_stamp = rospy.Time.from_sec(bb_ts)
        bb_list_msg.header.stamp = header_stamp
        for obj_id, (bb, class_str,valid) in processed_image.items():
            bb_msg = AngledBbox()
            bb_msg.header.stamp = header_stamp
            bb_msg.header.frame_id = '{}'.format(obj_id)  # this is an int defining which object this is
            bb_msg.x = bb[0]
            bb_msg.y = bb[1]
            bb_msg.width = bb[2]
            bb_msg.height = bb[3]
            bb_msg.angle = bb[4]
            bb_msg.im_seg_mode = bb_seg_mode
            bb_msg.class_str = class_str
            bb_msg.id = obj_id

            bb_list_msg.boxes.append(bb_msg)

        self.bb_data_pub.publish(bb_list_msg)

    def tracked_objects_poses_cb(self, msg, obj_name):
        self.latest_tracked_poses[obj_name] = msg.pose

    def get_closest_pose(self,class_str,pos):
        """
        Finds the closest object of a specific class to a given position using its groundtruth pose
        """
        max_dist = np.inf
        for obj_name in self.objects_names_per_class[class_str]:
            pose_obj = pose_msg_to_array(self.latest_tracked_poses[obj_name])
            dist = np.linalg.norm(pos-pose_obj[:3])
            if dist < max_dist:
                dist = max_dist
                pose = pose_obj
        return pose


