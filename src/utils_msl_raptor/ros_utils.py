####### ROS UTILITIES #######
# IMPORTS
# system
import pdb
# math
import numpy as np
import numpy.linalg as la
from bisect import bisect_left
# ros
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
try:
    from utils_msl_raptor.math_utils import *
except:
    from math_utils import *

def pose_to_tf(pose):
    """
    tf_w_q (w:world, q:quad) s.t. if a point is in the quad frame (p_q) then
    the point transformed to be in the world frame is p_w = tf_w_q * p_q.
    """
    tf_w_q = quat_to_tf([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
    tf_w_q[0:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])
    return tf_w_q


def pose_to_state_vec(pose):
    """ Turn a ROS pose message into a 13 el state vector (w/ 0 vels) """
    state = np.zeros((13,))
    state[0:3] = [pose.position.x, pose.position.y, pose.position.z]
    state[6:10] = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
    if state[6] < 0:
        state[6:10] *= -1
    return state


def tf_to_state_vec(tf):
    """ Turn a 4x4 tf pose message into a 13 el state vector (w/ 0 vels) """
    state = np.zeros((13,))
    state[0:3] = tf[0:3, 3]
    state[6:10] = rotm_to_quat(tf[0:3, 0:3])
    if state[6] < 0:
        state[6:10] *= -1
    return state


def get_ros_time(msg=None):
    """
    returns ros time currently or of a message in seconds 
    """
    if msg is None:
        return rospy.Time.now().to_sec()
    else:
        return msg.header.stamp.to_sec()


def find_closest_by_time(time_to_match, time_list, message_list=None):
    """
    Assumes lists are sorted earlier to later. Returns closest item in list by time. If two numbers are equally close, return the smallest number.
    Adapted from https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
    """
    if not message_list:
        message_list = time_list
    pos = bisect_left(time_list, time_to_match)
    if pos == 0:
        return message_list[0], 0
    if pos == len(time_list):
        return message_list[-1], len(message_list) - 1
    before = time_list[pos - 1]
    after = time_list[pos]
    if after - time_to_match < time_to_match - before:
       return message_list[pos], pos
    else:
       return message_list[pos - 1], pos - 1


def b_vs_debug():
    try:
        rospy.Time().now()
    except:
        return True
    return False
