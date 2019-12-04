####### MATH UTILITIES #######
# IMPORTS
# tools
import pdb
from copy import copy
# math
import numpy as np
import numpy.linalg as la
from pyquaternion import Quaternion
import tf


def enforce_quat_format(quat):
    """
    quat should have norm 1 and a positive first element (note orientation represented by q is same as -q)
    """
    unit_quat = quat / la.norm(quat)
    s = np.sign(unit_quat[0])
    if not s == 0:
        unit_quat *= s
    return unit_quat


def axang_to_quat(axang):
    """ 
    takes in an orientation in axis-angle form s.t. |axang| = ang, and 
    axang/ang = unit vector about which the angle is rotated. Returns a quaternion
    """
    ang = la.norm(axang)
    # deal with numeric issues if angle is 0
    if abs(ang) > 0.0001:
        quat = enforce_quat_format(Quaternion(axis=axang, angle=ang).elements)
    else:
        quat = np.array([1., 0., 0., 0.])
    return quat


def quat_to_axang(quat):
    """ 
    takes in an orientation in axis-angle form s.t. |axang| = ang, and 
    axang/ang = unit vector about which the angle is rotated. Returns a quaternion
    """
    q_obj = Quaternion(array=quat)  # turn a numpy array into a pyquaternion object
    return q_obj.radians * q_obj.axis  # will be [0, 0, 0] if no rotation


def quat_inv(q):
    """ technically this is the conjugate, for unit quats this is same as inverse """
    return enforce_quat_format(Quaternion(array=q).inverse.elements)


def quat_mul(q0, q1):
    """
    multiply q0 by q1. first element in quat is scalar value
    """
    return enforce_quat_format((Quaternion(array=q0) * Quaternion(array=q1)).elements)


def quat_to_ang(q):
    """
    Convert a quaternion to euler angles (ASSUMES 'XYZ')
    note: ros functions expect last element of quat to be scalar
    """
    roll, pitch, yaw = tf.transformations.euler_from_quaternion(np.array([q[1], q[2], q[3], q[0]]), 'rxyz')
    return roll, pitch, yaw


def ang_to_quat(angs):
    """
    Convert euler angles into a quaternion (ASSUMES 'XYZ')
    note: ros functions expect last element of quat to be scalar
    """
    q = tf.transformations.quaternion_from_euler(angs[0], angs[1], angs[2], 'rxyz')
    return np.array([q[3], q[0], q[1], q[2]])


def quat_to_rotm(quat):
    """ 
    calculate the rotation matrix of a given quaternion (frames assumed to be consistant 
    with the UKF state quaternion). First element of quat is the scalar.
    """
    return Quaternion(array=quat).rotation_matrix


def quat_to_tf(quat):
    """ 
    calculate the rotation matrix of a given quaternion (frames assumed to be consistant 
    with the UKF state quaternion). First element of quat is the scalar.
    """
    return Quaternion(array=quat).transformation_matrix
    

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

    return q_mean, ei_vec_set


def inv_tf(tf_in):
    # tf_inv = np.empty_like(tf)
    # tf_inv[3, 3] = 1
    # tf_inv[0:3, 0:3] = tf[0:3, 0:3].T
    # tf_inv[0:3, 3] = -tf_inv[0:3, 0:3] @ tf[0:3, 3]
    return tf.transformations.inverse_matrix(tf_in)
