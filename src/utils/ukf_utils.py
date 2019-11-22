####### UKF UTILITIES #######
# IMPORTS
# tools
import pdb
# math
import numpy as np
import numpy.linalg as la
import tf
# libs & utils
from utils.math_utils import *


def state_to_tf(state):
    tf = np.eye(4)
    tf[0:3, 3] = state[0:3]
    tf[0:3, 0:3] = quat_to_rotm(state[6:10])
    return tf


def enforce_pos_def_sym_mat(sigma):
    sigma_out = (sigma + sigma.T) / 2
    eig_vals, eig_vecs = la.eig(sigma_out)
    eig_val_mat = np.diag(np.real(eig_vals))
    eig_vec_mat = np.real(eig_vecs)
    eig_val_mat[eig_val_mat < 0] = 0.000001
    ### TEMP PYTHON 2 ###
    sigma_out = np.matmul(eig_vec_mat, np.matmul(eig_val_mat, eig_vec_mat.T))
    # sigma_out = eig_vec_mat @ eig_val_mat @ eig_vec_mat.T
    return sigma_out + 1e-12 * np.eye(sigma_out.shape[0])  # the small addition is for numeric stability
