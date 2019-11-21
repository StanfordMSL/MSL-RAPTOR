####### ROS UTILITIES #######
# IMPORTS
# math
import numpy as np
import numpy.linalg as la

def axang_to_quat(axang):
    """ 
    takes in an orientation in axis-angle form s.t. |axang| = ang, and 
    axang/ang = unit vector about which the angle is rotated. Returns a quaternion
    """
    quat = np.array([1, 0, 0, 0])
    ang = la.norm(axang)

    # deal with numeric issues
    if abs(ang) > 0.001:
        vec_perturb = axang / ang
        quat[0] = np.cos(ang/2)
        quat[1:4] = np.sin(ang/2) * vec_perturb
    
    return quat

def quat_mul(q0, q1):
    """
    multiply q0 by q1. first element in quat is scalar value
    """
    print("in quat_mul")
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                      x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
