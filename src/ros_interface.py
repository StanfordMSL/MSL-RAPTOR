# IMPORTS
# system
import sys, time #, argparse
# math
import numpy as np
from bisect import bisect_left
# ros
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf


class ros_interface:

    def __init__(self):
        
        self.VERBOSE = True

        # Paramters #############################
        self.b_detect_new_bb = True  # set to false if last frame we had a bb (if false, use a tracking network like SiamMask)
        self.latest_bb = None
        self.latest_ego_pose = None
        self.latest_bb_method = 1  # 1 for detect network, -1 for tracking network
        self.latest_time = 0
        self.image_dims = None
        self.pose_buffer = ([], [])
        self.pose_queue_size = 50
        self.start_time = 0
        self.quad_pose_gt = None
        ####################################################################


        self.ns = rospy.get_param('~ns')  # robot namespace
        
        # Subscribers / Listeners & Publishers #############################        
        camera_info = rospy.wait_for_message(self.ns + '/camera/camera_info', CameraInfo, 5)
        self.start_time = camera_info.header.stamp.to_sec()
        self.K = camera_info.K


        rospy.Subscriber(self.ns + '/camera/image_raw', Image, self.image_cb)
        rospy.Subscriber(self.ns + '/mavros/local_position/pose', PoseStamped, self.pose_ekf_cb, queue_size=10)  # internal ekf pose
        rospy.Subscriber(self.ns + '/mavros/vision_pose/pose', PoseStamped, self.pose_gt_cb, queue_size=10)  # optitrack pose
        self.state_pub = rospy.Publisher(self.ns + '/msl_raptor_state', Float32MultiArray, queue_size=10)
        ####################################################################
        

    def pose_ekf_cb(self, msg):
        """
        Maintains a buffer of poses and times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """

        time = self.get_ros_time()  # time in seconds

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
        if not self.image_dims:
            self.image_dims = (msg.height, msg.width)

        if len(self.pose_buffer[0]) == 0:
            return # this happens if we are just starting

        time = msg.header.stamp.to_sec() - self.start_time  # timestamp in seconds
        self.latest_ego_pose = self.find_closest_by_time(time)[0]

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
        data_len = len(state)
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
        self.state_pub.publish(state_msg)


    def get_ros_time(self):
        """
        returns ros time in seconds (minus time at start of program)
        """
        ts = rospy.Time.now()
        return ts.to_sec() - self.start_time


    def find_closest_by_time(self, time_to_match):
        """
        Assumes my_list is sorted. Returns closest item in list by time. If two numbers are equally close, return the smallest number.
        Adapted from https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
        """
        pose_list = self.pose_buffer[0]
        time_list = self.pose_buffer[1]
        pos = bisect_left(time_list, time_to_match)
        if pos == 0:
            return pose_list[0], 0
        if pos == len(myList):
            return pose_list[-1], len(pose_list) - 1
        before = time_list[pos - 1]
        after = time_list[pos]
        if after - time_to_match < time_to_match - before:
           return pose_list[pos], pos
        else:
           return pose_list[pos - 1], pos - 1


####### ROS UTILITIES #######
    def pose_to_tf(self, pose):
        """
        tf_w_q (w:world, q:quad) s.t. if a point is in the quad frame (p_q) then
        the point transformed to be in the world frame is p_w = tf_w_q * p_q.
        """
        tf_w_quad = tf.transformations.quaternion_matrix((pose.orientation.x, 
                                                          pose.orientation.y, 
                                                          pose.orientation.z, 
                                                          pose.orientation.w))
        tf_w_quad[0:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])
        return tf_w_quad


    def pose_to_state_vec(self, pose):
        """ Turn a ROS pose message into a 13 el state vector (w/ 0 vels) """
        state = np.zeros((13,))
        state[0:3] = [pose.position.x, pose.position.y, pose.position.z]
        state[6:10] = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        if state[6] < 0:
            state[6:10] *= -1
        return state
