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
from utils.ros_utils import *


class ros_interface:

    def __init__(self, b_DEBUG=False):
        
        self.VERBOSE = True

        # Paramters #############################
        self.b_detect_new_bb = True  # set to false if last frame we had a bb (if false, use a tracking network like SiamMask)
        self.latest_bb = None
        self.pose_ego_w = None
        self.latest_bb_method = 1  # 1 for detect network, -1 for tracking network
        self.latest_time = -1
        self.image_dims = None
        self.pose_buffer = ([], [])
        self.pose_queue_size = 50
        self.start_time = 0
        self.quad_pose_gt = None
        self.start_time = None
        ####################################################################

        # Subscribers / Listeners & Publishers #############################   
        self.ns = rospy.get_param('~ns')  # robot namespace
        rospy.Subscriber(self.ns + '/camera/image_raw', Image, self.image_cb)
        rospy.Subscriber(self.ns + '/mavros/local_position/pose', PoseStamped, self.pose_ekf_cb, queue_size=10)  # internal ekf pose
        rospy.Subscriber(self.ns + '/mavros/vision_pose/pose', PoseStamped, self.pose_gt_cb, queue_size=10)  # optitrack pose
        self.state_pub = rospy.Publisher(self.ns + '/msl_raptor_state', Float32MultiArray, queue_size=10)
        ####################################################################

        # DEBUGGGGGGGGG
        if b_DEBUG:
            self.tracked_quad_pose_gt = None
            rospy.Subscriber('/quad4' + '/mavros/vision_pose/pose', PoseStamped, self.debug_tracked_pose_gt_cb, queue_size=10)  # DEBUG ONLY - optitrack pose
        ##########################
    
    
    def debug_tracked_pose_gt_cb(self, msg):
        self.tracked_quad_pose_gt = msg.pose


    def pose_ekf_cb(self, msg):
        """
        Maintains a buffer of poses and times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """
        if self.start_time is None:
            return

        time = get_ros_time(self.start_time)  # time in seconds

        if len(self.pose_buffer[0]) < self.pose_queue_size:
            self.pose_buffer[0].append(msg.pose)
            self.pose_buffer[1].append(time)
        else:
            self.pose_buffer[0][0:self.pose_queue_size] = self.pose_buffer[0][1:self.pose_queue_size]
            self.pose_buffer[1][0:self.pose_queue_size] = self.pose_buffer[1][1:self.pose_queue_size]
            self.pose_buffer[0][-1] = msg.pose
            self.pose_buffer[1][-1] = time


    def pose_gt_cb(self, msg):
        self.quad_pose_gt = msg.pose


    def image_cb(self, msg):
        """
        recieve an image, process w/ NN, then set variables that the main function will access 
        note: set the time variable at the end (this signals the program when new data has arrived)
        """
        if self.start_time is None:
            self.start_time = msg.header.stamp.to_sec()
            time = 0
            self.image_dims = (msg.height, msg.width)
        else:
            time = msg.header.stamp.to_sec() - self.start_time  # timestamp in seconds

        if len(self.pose_buffer[0]) == 0:
            return # this happens if we are just starting

        self.pose_ego_w = find_closest_by_time(time, self.pose_buffer[1], self.pose_buffer[0])[0]

        # call NN here!!!!
        image = msg.data
        if self.b_detect_new_bb:
            self.latest_bb_method = 1
        else:
            self.latest_bb_method = -1
            
        self.latest_bb = [120, 230, 40, 20, 10*np.pi/180]
        self.latest_time = time  # DO THIS LAST


    def publish_filter_state(self, state):
        """
        Broadcast the estimated state of the filter. 
        State assumed to be a Nx1 numpy array of floats
        """
        MAX_VAL_ERR_THRESH = 1e10
        if np.any(state > MAX_VAL_ERR_THRESH):
            rospy.logwarn("VALUES IN STATE ARE TOO BIG TO BE RIGHT!! Truncating to avoid error")
            state[state > MAX_VAL_ERR_THRESH] = MAX_VAL_ERR_THRESH
        data_len = state.shape[0]
        state_msg = Float32MultiArray()
        state_msg.layout.dim.append(MultiArrayDimension())
        state_msg.layout.dim.append(MultiArrayDimension())
        state_msg.layout.dim[0].size = data_len
        state_msg.layout.dim[1].size = 1
        state_msg.layout.dim[0].stride = data_len*1
        state_msg.layout.dim[1].stride = data_len
        state_msg.layout.dim[0].label = "rows"
        state_msg.layout.dim[1].label = "cols"
        state_msg.layout.data_offset = 0
        state_msg.data = list(state)
        try:
            self.state_pub.publish(state_msg)
        except Exception as e:
            print("failed pub (ros_interface): {}".format(e))
            if not b_vs_debug():
                pdb.set_trace()


