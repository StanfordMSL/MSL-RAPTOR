####### MATH UTILITIES #######
# IMPORTS
# tools
import pdb
from copy import copy
# math
import numpy as np
import numpy.linalg as la
import tf
from scipy.spatial.transform import Rotation as R

def enforce_quat_format(quat):
    """
    quat should have norm 1 and a positive first element (note orientation represented by q is same as -q)
    input: nx4 array
    """
    quat = quat.reshape(-1,4)
    unit_quat = quat / la.norm(quat,axis=1).reshape(-1,1)
    s = np.sign(unit_quat[:,0])
    # Not change the quaternion when the scalar is 0
    s = ( s== 0) * 1 + s
    unit_quat *= s.reshape(-1,1)
    return unit_quat


def axang_to_quat(axang):
    """ 
    takes in an orientation in axis-angle form s.t. |axang| = ang, and 
    axang/ang = unit vector about which the angle is rotated. Returns a quaternion
    """
    axang = axang.reshape(-1,3)
    quat = np.roll(R.from_rotvec(axang).as_quat(),1,axis=1)
    return enforce_quat_format(quat)


def quat_to_axang(quat):
    """ 
    takes in an orientation in axis-angle form s.t. |axang| = ang, and 
    axang/ang = unit vector about which the angle is rotated. Returns a quaternion
    """
    return R.from_quat(np.roll(quat,3,axis=1)).as_rotvec()


def quat_inv(q):
    """ technically this is the conjugate, for unit quats this is same as inverse """
    q_inv = np.array(q, copy=True)
    if q_inv[0] > 0:
        q_inv[1:4] *= -1
    else:
        q_inv[0] *= -1
    return q_inv


def quat_mul(q, r):
    """
    multiply q by r. first element in quat is scalar value. Can be nx4 sized numpy arrays
    """
    q = q.reshape(-1,4)
    r = r.reshape(-1,4)
    vec = np.array([np.multiply(q[:,0],r[:,1]),np.multiply(q[:,0],r[:,2]),np.multiply(q[:,0],r[:,3])]).T + np.array([np.multiply(r[:,0],q[:,1]),np.multiply(r[:,0],q[:,2]),np.multiply(r[:,0],q[:,3])]).T + np.array([np.multiply(q[:,2],r[:,3])-np.multiply(q[:,3],r[:,2]),np.multiply(q[:,3],r[:,1])-np.multiply(q[:,1],r[:,3]),np.multiply(q[:,1],r[:,2])-np.multiply(q[:,2],r[:,1])] ).T
    scalar = (np.multiply(q[:,0] ,r[:,0]) - np.sum(np.multiply(q[:,1:],r[:,1:]),axis=1)).reshape(-1,1)
    qout = np.concatenate((scalar,vec),axis=1)
    return enforce_quat_format(qout)

def quat_to_ang(q):
    """
    Convert a quaternion to euler angles (ASSUMES 'XYZ')
    """
    return R.from_quat(np.roll(q,3,axis=1)).as_euler('XYZ')

def ang_to_quat(angs):
    """
    Convert euler angles into a quaternion (ASSUMES 'XYZ')
    """
    return np.roll(R.from_euler('XYZ',angs).as_quat(),1,axis=1)


def quat_to_rotm(quat):
    """ 
    calculate the rotation matrix of a given quaternion (frames assumed to be consistant 
    with the UKF state quaternion). First element of quat is the scalar.
    """
    return R.from_quat(np.roll(np.reshape(quat, (-1, 4)),3,axis=1)).as_dcm()


def quat_to_tf(quat):
    """ 
    calculate the rotation matrix of a given quaternion (frames assumed to be consistant 
    with the UKF state quaternion). First element of quat is the scalar.
    """
    tf_out = np.eye(4)
    tf_out[0:3, 0:3] = quat_to_rotm(quat)
    return tf_out
    

def average_quaternions(Q, w=None):
    """
    Adapted from https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py
    Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
    The quaternions are arranged as (w,x,y,z), with w being the scalar
    The result will be the average quaternion of the input. Note that the signs
    of the output quaternion can be reversed, since q and -q describe the same orientation
    The weight vector w must be of the same length as the number of rows in the quaternion maxtrix Q
    """
    M = Q.shape[0]  # Number of quaternions to average
    
    if w is None:
        w = np.ones((M,)) / M  # DEFAULT: equally weighted

    A = np.zeros((4,4))
    weightSum = 0

    for i in range(0,M):
        q = Q[i, :]
        A = w[i] * np.outer(q, q) + A
        weightSum += w[i]

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = la.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    q_mean = np.real(eigenVectors[:, 0])
    if q_mean[0] < 0:
        q_mean *= -1
        
    # calc set of differences from each quat to the mean (in ax-angle rep.)
    ei_vec_set = np.zeros((3, M)) 
    q_mean_inv = quat_inv(q_mean)
    for i, qi in enumerate(Q):
        ei_quat = quat_mul(qi, q_mean_inv)
        ei_vec_set[:, i] = quat_to_axang(ei_quat)

    return enforce_quat_format(q_mean), ei_vec_set


def inv_tf(tf_in):
    return tf.transformations.inverse_matrix(tf_in)
