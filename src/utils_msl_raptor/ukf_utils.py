####### UKF UTILITIES #######
# IMPORTS
# tools
import pdb
# math
import numpy as np
import numpy.linalg as la
import tf
import cv2
# libs & utils
try:
    from utils_msl_raptor.math_utils import *
except:
    from math_utils import *
from os import listdir
import yaml 

def state_to_tf(state):
    """ returns tf_w_quad given the state vector """
    tf_w_quad = np.eye(4)
    tf_w_quad[0:3, 3] = state[0:3]
    tf_w_quad[0:3, 0:3] = quat_to_rotm(state[6:10])
    return tf_w_quad


def enforce_pos_def_sym_mat(sigma):
    return nearestPD(sigma)
    # sigma_out = (sigma + sigma.T) / 2
    # eig_vals, eig_vecs = la.eig(sigma_out)
    # eig_val_mat = np.diag(np.real(eig_vals))
    # eig_vec_mat = np.real(eig_vecs)
    # eig_val_mat[eig_val_mat < 0] = 0.000001
    # sigma_out = eig_vec_mat @ eig_val_mat @ eig_vec_mat.T
    # return sigma_out + 1e-10 * np.eye(sigma_out.shape[0])  # the small addition is for numeric stability


######################################################################################################
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
######################################################################################################

def bb_corners_to_angled_bb(points, output_coord_type='xy'):
    """
    BB_CORNERS_TO_ANGLE Function that takes takes coordinates of a bounding
    box corners and returns it as center, size and angle.
    points = [x1,y1;x2,y2;x3,y3;x4,y4] (N x 2 matrix)
    """
    sortind_x = np.argsort(points[:, 0], axis=0)  # sort points by x coordinate

    # of the points furthest to the left, which is lower and which is higher? (bottom left / top left)
    if points[sortind_x[0], 1] > points[sortind_x[1], 1]:
        bl_x = points[sortind_x[0], 0]
        bl_y = points[sortind_x[0], 1]
        tl_x = points[sortind_x[1], 0]
        tl_y = points[sortind_x[1], 1]
    else:
        bl_x = points[sortind_x[1], 0]
        bl_y = points[sortind_x[1], 1]
        tl_x = points[sortind_x[0], 0]
        tl_y = points[sortind_x[0], 1]

    # of the points furthest to the right, which is lower and which is higher? (bottom right / top right)
    if points[sortind_x[2], 1] > points[sortind_x[3], 1]:
        br_x = points[sortind_x[2], 0]
        br_y = points[sortind_x[2], 1]
        tr_x = points[sortind_x[3], 0]
        tr_y = points[sortind_x[3], 1]
    else:
        br_x = points[sortind_x[3], 0]
        br_y = points[sortind_x[3], 1]
        tr_x = points[sortind_x[2], 0]
        tr_y = points[sortind_x[2], 1]

    # print(bl_x)
    # print(bl_y)
    # print(br_x)
    # print(br_y)
    if np.abs(br_x - bl_x) < 0.00001:
        print("error")
    angle = -np.arctan((bl_y - br_y) / (bl_x - br_x))
    x_center = np.mean([br_x, bl_x, tl_x, tr_x])
    y_center = np.mean([br_y, bl_y, tl_y, tr_y])
    width = la.norm([br_x - bl_x, br_y - bl_y])
    height = la.norm([br_x - tr_x, br_y - tr_y])
    output = np.array([x_center, y_center, width, height, angle])
    if output_coord_type.lower() == 'rc':
        # r is y, col is x
        output = np.array([y_center, x_center, width, height, angle])

    return output

def remove_roll(quats):
    quats = quats.reshape(-1,4)
    angles= quat_to_ang(quats)
    angles[:,0] = 0
    return ang_to_quat(angles)

def remove_pitch(quats):
    quats = quats.reshape(-1,4)
    angles= quat_to_ang(quats)
    angles[:,1] = 0
    return ang_to_quat(angles)
    
def remove_yaw(quats):
    quats = quats.reshape(-1,4)
    angles= quat_to_ang(quats)
    angles[:,2] = 0
    return ang_to_quat(angles)


def enforce_constraints_pose(pose, b_enforce_0_dict, fixed_vals):
    """
    Takes in a pose (from a ros message) and projects it to satisfy the contraints
    """
    if len(b_enforce_0_dict) == 0:
        return pose

    # handle position
    if b_enforce_0_dict['x']:
        if 'x' in fixed_vals:
            pose.position.x = fixed_vals['x']
        else:
            pose.position.x = 0
    if b_enforce_0_dict['y']:
        if 'y' in fixed_vals:
            pose.position.y = fixed_vals['y']
        else:
            pose.position.y = 0
    if b_enforce_0_dict['z']:
        if 'z' in fixed_vals:
            pose.position.z = fixed_vals['z']
        else:
            pose.position.z = 0

    # handle orientation
    quat = np.reshape([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z], (1,4))
    if b_enforce_0_dict['roll']:
        quat = remove_roll(quat)
    if b_enforce_0_dict['pitch']:
        quat = remove_pitch(quat)
    if b_enforce_0_dict['yaw']:
        quat = remove_yaw(quat)

    pose.orientation.w = quat[0, 0]
    pose.orientation.x = quat[0, 1]
    pose.orientation.y = quat[0, 2]
    pose.orientation.z = quat[0, 3]

    return pose


def pose_to_3d_bb_proj(tf_w_ado, tf_w_ego, vertices_ado, camera):
    """
    vertices_ado (ado frame) is a N x 4 where the 4 [x, y, z, 1; ...] matrix
    """
    N = vertices_ado.shape[0]
    tf_cam_ado = camera.tf_cam_ego @ inv_tf(tf_w_ego) @ tf_w_ado
    vertices_cam = tf_cam_ado @ vertices_ado.T
    projected_vertices = np.zeros((N, 2))
    for i, bb_vert in enumerate(vertices_cam.T):
        projected_vertices[i, :] = camera.pnt3d_to_pix(bb_vert)

    return projected_vertices

def load_category_params():
    params = {}
    for f in listdir("/root/msl_raptor_ws/src/msl_raptor/params/category_params"):
        class_str = f.split('_')[0]
        yaml_file = class_str+'_ukf_params.yaml'

        with open( "/root/msl_raptor_ws/src/msl_raptor/params/category_params/"+ f) as stream:
            try:
                params[class_str] = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    return params

def verts_to_angled_bb(verts, measurement=None):
    """
    verts should be [[x,y],[x,y]...]
    """
    # construct sensor output
    # minAreaRect sometimes flips the w/h and angle from how we want the output to be
    # to fix this, we can use boxPoints to get the x,y of the bb rect, and use our function
    # to get the output in the form we want 
    rect = cv2.minAreaRect(verts.astype('float32'))  # apparently float64s cause this function to fail
    box = cv2.boxPoints(rect)
    output = bb_corners_to_angled_bb(box, output_coord_type='xy')
    if measurement is not None:
        ang_thesh = np.deg2rad(20)  # angle threshold for considering alternative box rotation
        alt_ang = -np.sign(output[-1]) * (np.pi/2 - np.abs(output[-1]))  # negative complement of angle
        
        if (abs(alt_ang - measurement[-1]) < abs(output[-1] - measurement[-1])) and np.abs((np.abs(measurement[-1] - output[-1]) - np.pi/2)) < ang_thesh:
            output[-1] = alt_ang 
            w = output[2]
            h = output[3]
            output[2] = h
            output[3] = w
    return output
