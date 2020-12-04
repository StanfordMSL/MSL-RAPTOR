####### ROS UTILITIES #######
# IMPORTS
# system
import pdb
import os
import yaml
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

def pose_msg_to_array(pose):
    """
    returns [x,y,z,qw,qx,qy,qz]
    """
    return [pose.position.x, pose.position.y, pose.position.z, pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]


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
    if time_list is None or len(time_list)==0:
        raise RuntimeError("missing time info!")
    if message_list is None:
        message_list = time_list
    pos = bisect_left(time_list, time_to_match)
    if pos == 0:
        return message_list[0], 0
    if pos == len(time_list):
        return message_list[-1], len(message_list) - 1

    if pos - 1 < 0:
        return message_list[0], 0
    elif pos - 1 >= len(time_list):
        return time_list[-1], len(time_list)
        
    before = time_list[pos - 1]
    after = time_list[pos]
    if after - time_to_match < time_to_match - before:
       return message_list[pos], pos
    else:
       return message_list[pos - 1], pos - 1


def update_running_average(ave_info, new_el):
    """
    ave_info: [running mean, # of elements so far (not counting new one)]
    new_el: new element to be put into running mean
    """
    if ave_info[0] == 0:
        ave_info = [ave_info[-1], 1]
    else:
        ave_info[0] = ave_info[0] + (new_el - ave_info[0]) / ave_info[1]
        ave_info[1] += 1
    return ave_info


def get_object_sizes_from_yaml(objects_sizes_yaml, objects_used_path, classes_names_file, category_params):
    # create camera object (see https://github.com/StanfordMSL/uav_game/blob/tro_experiments/ec_quad_sim/ec_quad_sim/param/quad3_trans.yaml)

    with open(objects_used_path) as f:
        objects_used = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    objects_used = [x.strip() for x in objects_used]

    with open(classes_names_file) as f:
        classes_names = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    classes_names = [x.strip() for x in classes_names]
    classes_used_names = []
    classes_used_ids = []
    bb_3d = {}
    obj_width = {}
    obj_height = {}
    conneted_inds = {}

    objects_names_per_class = {}
    with open(objects_sizes_yaml, 'r') as stream:
        try:
            obj_prms = list(yaml.load_all(stream))
            for obj_dict in obj_prms:
                if obj_dict['ns'] in objects_used:
                    if 'cust_vert_file' in obj_dict and obj_dict['cust_vert_file'] and not obj_dict['cust_vert_file'] == "":
                        # If we are here then a custom vertex file was provided (instead of just height / width/ length for a box)
                        #   first load in the verts, then check if there is a list of pairs of verts to use when drawing the volume
                        print("USING CUSTOM VERTS!!! (for {})".format(obj_dict['ns']))
                        file_path = obj_dict['cust_vert_file'] + obj_dict['ns']
                        loaded_verts = np.loadtxt(file_path)

                        loaded_verts *= np.reshape(obj_dict['cust_vert_scale'], (1,3)).astype(float)

                        Angle_x = float(obj_dict['cust_vert_rpy'][0])
                        Angle_y = float(obj_dict['cust_vert_rpy'][1])
                        Angle_z = float(obj_dict['cust_vert_rpy'][2])
                        R_deltax = np.array([[ 1.             , 0.             , 0.              ],
                                                [ 0.             , np.cos(Angle_x),-np.sin(Angle_x) ],
                                                [ 0.             , np.sin(Angle_x), np.cos(Angle_x) ]])
                        R_deltay = np.array([[ np.cos(Angle_y), 0.             , np.sin(Angle_y) ],
                                                [ 0.             , 1.             , 0               ],
                                                [-np.sin(Angle_y), 0.             , np.cos(Angle_y) ]])
                        R_deltaz = np.array([[ np.cos(Angle_z),-np.sin(Angle_z), 0.              ],
                                                [ np.sin(Angle_z), np.cos(Angle_z), 0.              ],
                                                [ 0.             , 0.             , 1.              ]])
                        R_delta = R_deltax @ R_deltay @ R_deltaz
                        # R_delta = np.eye(3)

                        loaded_verts = (R_delta @ loaded_verts.T).T
                        bb_3d[obj_dict['class_str']] = np.concatenate(( loaded_verts, np.ones((loaded_verts.shape[0],1)) ), axis=1)

                        half_width = (np.max(bb_3d[obj_dict['class_str']][:, 0]) - np.min(bb_3d[obj_dict['class_str']][:, 0])) / 2
                        half_height = (np.max(bb_3d[obj_dict['class_str']][:, 2]) - np.min(bb_3d[obj_dict['class_str']][:, 2])) / 2

                        file_path += "_joined_inds"
                        if os.path.exists(file_path):
                            print(obj_dict['ns'])
                            print(obj_dict['class_str'])
                            conneted_inds[obj_dict['class_str']] = np.loadtxt(file_path).astype(int)
                        else:
                            conneted_inds[obj_dict['class_str']] = None
                    else:
                        half_length = (float(obj_dict['bound_box_l']) + category_params[obj_dict['class_str']]['offset_bb_l']) /2
                        half_width = (float(obj_dict['bound_box_w']) + category_params[obj_dict['class_str']]['offset_bb_w']) /2 
                        half_height = (float(obj_dict['bound_box_h']) + category_params[obj_dict['class_str']]['offset_bb_h'])/2
                        
                        bb_3d[obj_dict['class_str']] = np.array([[ half_length, half_width, half_height, 1.],  # 1 front, left,  up (from quad's perspective)
                                                                 [ half_length, half_width,-half_height, 1.],  # 2 front, right, up
                                                                 [ half_length,-half_width,-half_height, 1.],  # 3 back,  right, up
                                                                 [ half_length,-half_width, half_height, 1.],  # 4 back,  left,  up
                                                                 [-half_length,-half_width, half_height, 1.],  # 5 front, left,  down
                                                                 [-half_length,-half_width,-half_height, 1.],  # 6 front, right, down
                                                                 [-half_length, half_width,-half_height, 1.],  # 7 back,  right, down
                                                                 [-half_length, half_width, half_height, 1.]]) # 8 back,  left,  down

                    # Rescale objects
                    bb_3d[obj_dict['class_str']][:,:3], half_width,half_height = scale_3d_points(bb_3d[obj_dict['class_str']][:,:3], half_width,half_height,category_params[obj_dict['class_str']]['scales_xyz'])

                    obj_width[obj_dict['class_str']] = 2*half_width
                    obj_height[obj_dict['class_str']] = 2*half_height

                    # Add the object's class to the classes used, if not there already
                    if obj_dict['class_str'] not in classes_names:
                        print('Class '+obj_dict['class_str']+' not found in the list of classes supported')
                    elif obj_dict['class_str'] not in classes_used_names:
                        classes_used_names.append(obj_dict['class_str'])
                        classes_used_ids.append(classes_names.index(obj_dict['class_str']))
                        objects_names_per_class[obj_dict['class_str']] = [obj_dict['ns']]
                    else:
                        objects_names_per_class[obj_dict['class_str']].append(obj_dict['ns'])


        except yaml.YAMLError as exc:
            print(exc)

    return bb_3d, obj_width, obj_height, classes_used_names, classes_used_ids, objects_names_per_class, conneted_inds