####### MATH UTILITIES #######
# IMPORTS
# tools
import pdb
# math
import numpy as np
import numpy.linalg as la
import tf


def axang_to_quat(axang):
    """ 
    takes in an orientation in axis-angle form s.t. |axang| = ang, and 
    axang/ang = unit vector about which the angle is rotated. Returns a quaternion
    """
    quat = np.array([1., 0., 0., 0.])
    ang = la.norm(axang)

    # deal with numeric issues
    if abs(ang) > 0.001:
        vec_perturb = axang / ang
        quat[0] = np.cos(ang/2)
        quat[1:4] = np.sin(ang/2) * vec_perturb
    
    return quat


def quat_to_axang(quat):
    """ 
    takes in an orientation in axis-angle form s.t. |axang| = ang, and 
    axang/ang = unit vector about which the angle is rotated. Returns a quaternion
    """
    axang = np.zeros((3,))
    # deal with some numerical issues...
    if abs(quat[0]) > 1:
        if abs(la.norm(quat) - 1) < 0.0001:
            # this is a numeric issue (w/ no rotation)
            axang = np.zeros((3,))
        else:
            raise RuntimeException("Invalid quaternion!")

    # handle numeric issues with arccos and quaternion ( [-1, 1] <==> [pi, 0] )
    eps = 0.00001
    if 1 < quat[0] and quat[0] < 1 + eps:
        ang = 0
    elif -1 > quat[0] and quat[0] > -1 -eps:
        ang = np.pi
    elif -1 - eps < quat[0] and quat[0] < 1 + eps:
        ang = 2 * np.arccos(quat[0])  # this should be the case most of the time
    else:
        raise RuntimeException("Should never be here! issue with arccos(quat[0]), quat[0] = {:.8f}".format(quat[0]))

    if abs(ang) < 0.0001:
        return axang  # no rotation, return zeros

    ax = quat[1:4] / la.norm(quat[1:4])  # unit vector
    return ang * ax


def quat_inv(q):
    """ technically this is the conjugate, for unit quats this is same as inverse """
    return np.array([-q[0], q[1], q[2], q[3]])


def quat_mul(q0, q1):
    """
    multiply q0 by q1. first element in quat is scalar value
    """
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    q_out = np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                       x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                      -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    if q_out[0] < 0:
        q_out *= -1
    return q_out

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
    return tf.transformations.quaternion_matrix((quat[0], quat[1], quat[2], quat[3]))[0:3, 0:3]
    

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