####### ROS UTILITIES #######
# IMPORTS
# math
import numpy as np
# ros
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf


def pose_to_tf(pose):
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


def pose_to_state_vec(pose):
    """ Turn a ROS pose message into a 13 el state vector (w/ 0 vels) """
    state = np.zeros((13,))
    state[0:3] = [pose.position.x, pose.position.y, pose.position.z]
    state[6:10] = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    if state[6] < 0:
        state[6:10] *= -1
    return state
