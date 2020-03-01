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
import random

class ros_interface:

    def __init__(self, b_use_gt_bb=False,b_verbose=False,b_use_gt_pose_init=False,b_use_gt_detect_bb=False,b_pub_3d_bb_proj=False):
        
        self.verbose = b_verbose

        # Parameters #############################
        self.im_process_output = []  # what is accessed by the main function after an image is processed

        self.ego_pose_rosmesg_buffer = ([], [])
        self.ego_pose_rosmesg_buffer_len = 50
        self.ego_pose_gt_rosmsg = None

        self.im_seg = None  # object for parsing images into angled bounding boxes
        self.b_use_gt_bb = b_use_gt_bb  # toggle for debugging using ground truth bounding boxes
        self.latest_img_time = -1
        self.front_end_time = None
        ####################################################################

        self.ns = rospy.get_param('~ns')  # robot namespace

        self.bridge = CvBridge()
        self.camera = None

        # DEBUGGGGGGGGG
        if True or b_use_gt_bb:
            self.ado_pose_gt_rosmsg = None
            rospy.Subscriber('/quad4' + '/mavros/vision_pose/pose', PoseStamped, self.ado_pose_gt_cb, queue_size=10)  # DEBUG ONLY - optitrack pose
        ##########################

        # Just used for initializing from groundtruth
        self.objects_names_per_class = None
        self.bb_3d = None
        self.b_use_gt_pose_init  = b_use_gt_pose_init
        self.b_use_gt_detect_bb = b_use_gt_detect_bb
        self.b_pub_3d_bb_proj = b_pub_3d_bb_proj
        self.num_imgs_processed = 0


    def get_first_image(self):
        return self.bridge.imgmsg_to_cv2(rospy.wait_for_message(self.ns + '/camera/image_raw',Image), desired_encoding="bgr8")

    def create_subs_and_pubs(self):
        # Subscribers / Listeners & Publishers #############################   
        rospy.Subscriber(self.ns + '/camera/image_raw', Image, self.image_cb, queue_size=1,buff_size=2**21)
        rospy.Subscriber(self.ns + '/mavros/local_position/pose', PoseStamped, self.ego_pose_ekf_cb, queue_size=10)  # internal ekf pose
        rospy.Subscriber(self.ns + '/mavros/vision_pose/pose', PoseStamped, self.ego_pose_gt_cb, queue_size=10)  # optitrack pose
        self.state_pub = rospy.Publisher(self.ns + '/msl_raptor_state', TrackedObjects, queue_size=5)
        self.bb_data_pub = rospy.Publisher(self.ns + '/bb_data', AngledBboxes, queue_size=5)

        if self.b_use_gt_pose_init or self.b_use_gt_detect_bb:
            # Create dict to store pose for each object
            self.latest_tracked_poses = {}
            for obj_name in sum(self.objects_names_per_class.values(),[]):
                rospy.Subscriber(obj_name + '/mavros/vision_pose/pose', PoseStamped, self.tracked_objects_poses_cb, obj_name,queue_size=5)
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

        t_fe_start = time.time()  # start timer for frontend

        self.tf_w_ego = pose_to_tf(find_closest_by_time(my_time, self.ego_pose_rosmesg_buffer[1], self.ego_pose_rosmesg_buffer[0])[0])

        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
        
        # undistort the fisheye effect in the image
        if self.camera is not None:
            image = cv2.undistort(image, self.camera.K, self.camera.dist_coefs, None, self.camera.new_camera_matrix)
        
        self.latest_bb_method = self.im_seg.mode
        if self.b_use_gt_detect_bb:
            gt_bbs = self.get_gt_boxes()
            self.im_process_output = self.im_seg.process_image(image,my_time,gt_bbs)
        else:
            self.im_process_output = self.im_seg.process_image(image,my_time)
        self.front_end_time = time.time() - t_fe_start
        self.num_imgs_processed += 1
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

            if self.b_pub_3d_bb_proj:
                tmp = ukf_dict[id].projected_3d_bb
                tmp = tmp.reshape((tmp.size, ))
                obj.projected_3d_bb = tmp

                if ukf_dict[id].connected_inds is not None:
                    tmp = ukf_dict[id].connected_inds
                    tmp = tmp.reshape((tmp.size, ))
                    obj.connected_inds = tmp
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
                max_dist = dist
                pose = pose_obj
                obj_name_kept = obj_name
        print("Initialized tracking based on "+obj_name_kept+' - dist '+str(dist))
        return pose


    def get_gt_boxes(self):
        gt_boxes = []
        for class_str, obj_names in self.objects_names_per_class.items():
            for obj_name in obj_names:
                pose = pose_msg_to_array(self.latest_tracked_poses[obj_name])
                tf_w_ado = quat_to_tf(pose[3:])
                tf_w_ado[:3,3] = pose[:3]
                proj_corners = pose_to_3d_bb_proj(tf_w_ado,self.tf_w_ego,self.bb_3d[class_str],self.camera)
                (x,y,w,h) = corners_to_aligned_bb(proj_corners)

                # Add 5% of size noise
                dx = 0.05*random.uniform(-w,w)/2.
                dy = 0.05*random.uniform(-h,h)/2.
                dw = 0.05*random.uniform(-w,w)/2.
                dh = 0.05*random.uniform(-h,h)/2.

                gt_boxes.append([x+dx,y+dy,w+dw,h+dh,1.,1.,self.im_seg.class_str_to_id[class_str]])
        return np.array(gt_boxes)

