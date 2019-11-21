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
        quat = [np.cos(ang/2), np.sin(ang/2) * vec_perturb]
    
    return quat