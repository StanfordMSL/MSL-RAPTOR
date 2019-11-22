
# IMPORTS
#tools
from copy import copy
import pdb
import cv2
# math
import numpy as np
import numpy.linalg as la
# plots
# import matplotlib
# from matplotlib import pyplot as plt
# from mpl_toolkits import mplot3d
# ros
import rospy
# libs & utils
from utils.ukf_utils import *
from utils.math_utils import *


class UKF:

    def __init__(self):

        self.VERBOSE = True

        # Paramters #############################
        self.b_enforce_0_yaw = True;
        self.dim_state = 13
        self.dim_sig = 12  # covariance is 1 less dimension due to quaternion
        self.dim_meas = 5  # angled bounding box: row, col, width, height, angle

        alpha = .1  # scaling param - how far sig. points are from mean
        kappa = 2  # scaling param - how far sig. points are from mean
        beta = 2  # optimal choice according to probablistic robotics (textbook)
        ukf_lambda = alpha**2 * (self.dim_sig + kappa) - self.dim_sig
        self.sig_pnt_multiplier = np.sqrt(self.dim_sig + ukf_lambda)

        self.w0_m = ukf_lambda / (ukf_lambda + self.dim_sig)
        self.w0_c = self.w0_m + (1 - alpha**2 + beta)
        self.wi = 1 / (2 * (ukf_lambda + self.dim_sig))
        self.w_arr_mean = np.ones((1+ 2 * self.dim_sig,)) * self.wi
        self.w_arr_mean[0] = self.w0_m
        self.w_arr_cov = np.ones((1+ 2 * self.dim_sig,)) * self.wi
        self.w_arr_cov[0] = self.w0_c

        self.camera = None
        ####################################################################

        # init vars #############################
        self.ukf_itr = 0
        dp = 0.1  # [m]
        dv = 0.005  # [m/s]
        dq = 0.1  # [rad] in ax ang 
        dw = 0.005  # [rad/s]
        self.mu = np.zeros((self.dim_state, 1))
        self.sigma = np.diag([dp, dp, dp, dv, dv, dv, dq, dq, dq, dw, dw, dw])

        self.Q = self.sigma/10  # Process Noise
        self.R = np.diag([2, 2, 10, 10, 0.08])  # Measurement Noise
        ####################################################################


    def step_ukf(self, measurement, bb_3d, tf_ego_w, dt):
        """
        UKF iteration following pseudo code from probablistic robotics
        """
        rospy.loginfo("Starting UKF Iteration {}".format(self.ukf_itr))

        # line 2
        sps = self.calc_sigma_points(self.mu, self.sigma)

        # lines 2 & 3
        sps_prop = np.empty_like(sps)
        for sp_ind in range(sps.shape[1]):
            sps_prop[:, sp_ind] = self.propagate_dynamics(sps[:, sp_ind], dt)

        # lines 4 & 5
        mu_bar, sig_bar = self.extract_mean_and_cov_from_state_sigma_points(sps_prop)
        
        # line 6
        sps_recalc = self.calc_sigma_points(mu_bar, sig_bar)
        
        # lines 7 & 8
        pred_meas = np.zeros((self.dim_meas, sps.shape[1]))
        for sp_ind in range(sps.shape[1]):
            pred_meas[:, sp_ind] = self.predict_measurement(sps_recalc[:, sp_ind], bb_3d, tf_ego_w)
        self.ukf_itr += 1

        # sigma = enforce_pos_def_sym_mat(sigma)  # project sigma to pos. def. cone to avoid numeric issues


    def predict_measurement(self, state, bb_3d_quad, tf_ego_w):
        """
        use camera model & relative states to predict the angled 2d 
        bounding box seen by the ego drone 
        """

        # get relative transform from camera to quad
        tf_w_quad = state_to_tf(state)
        
        tf_cam_quad = np.matmul(self.camera.tf_cam_ego, np.matmul(tf_ego_w, tf_w_quad)) ### TEMP PYTHON 2 ###
        # tf_cam_quad = self.camera.tf_cam_ego @ tf_ego_w @ tf_w_quad
        
        # tranform 3d bounding box from quad frame to camera frame
        bb_3d_cam = np.matmul(tf_cam_quad, bb_3d_quad.T)[0:3, :].T ### TEMP PYTHON 2 ###
        # bb_3d_cam = (tf_cam_quad @ bb_3d_quad)[:, 0:3]

        # transform each 3d point to a 2d pixel (row, col)
        bb_rc_list = np.zeros((bb_3d_cam.shape[0], 2))
        for i, bb_vert in enumerate(bb_3d_cam):
            bb_rc_list[i, :] = self.camera.pnt3d_to_pix(bb_vert)

        # construct sensor output
        (xx, yy), (width, height), angle = cv2.minAreaRect(np.fliplr(bb_rc_list).astype('float32'))  # apparently float64s cause this function to fail
        angle *= np.pi/180
        r_center = yy + np.sin(angle) * height / 2  # row is the height (y) dim
        c_center = xx + np.cos(angle) * width / 2 # col is the width (x) dim
        return np.array([r_center, c_center, width, height, angle*np.pi/180])


    
    def calc_sigma_points(self, mu, sigma):
        sps = np.zeros((self.dim_state, 2 * self.dim_sig + 1))
        sps[:, 0] = mu
        sig_step = self.sig_pnt_multiplier * la.cholesky(sigma)

        if self.b_enforce_0_yaw:
            sig_step[8, :] = 0

        q_nom = mu[6:10]
        # loop over half (-1) the num sigma points and update two at once
        for sp_ind in range(self.dim_sig):
            # first step in positive direction
            sp_col_1 = 1 + 2 * sp_ind  # starting in the second col, count by pairs
            sps[0:6, sp_col_1] = mu[0:6] + sig_step[0:6, sp_ind]
            sps[10:, sp_col_1] = mu[10:13] + sig_step[9:12, sp_ind]
            q_perturb = axang_to_quat(sig_step[6:9, sp_ind])
            sps[6:10, sp_col_1] = quat_mul(q_perturb, q_nom)

            # next step in positive direction
            sp_col_2 = sp_col_1 + 1
            sps[0:6, sp_col_2] = mu[0:6] - sig_step[0:6, sp_ind]
            sps[10:, sp_col_2] = mu[10:13] - sig_step[9:12, sp_ind]
            q_perturb = axang_to_quat(-sig_step[6:9, sp_ind])
            sps[6:10, sp_col_2] = quat_mul(q_perturb, q_nom)
            
        return sps
    
    
    def propagate_dynamics(self, state, dt):
        """
        Estimate the next state vector. Assumes no control input (velocities stay the same)
        """
        next_state = copy(state)

        # update position
        next_state[0:3] += dt * state[3:6]

        # update orientation
        quat = state[6:10]  # current orientation
        omegas = state[10:13]  # angular velocity vector
        om_norm = la.norm(omegas)  # rate of change of all angles
        if om_norm > 0:
            ang = om_norm * dt  # change in angle in this small timestep
            ax = omegas / om_norm  # axis about angle change
            quat_delta = axang_to_quat(ax * ang)
            quat_new = quat_mul(quat_delta, quat)
        else:
            quat_new = quat

        [roll, pitch, yaw] = quat_to_ang(quat_new)
        if self.b_enforce_0_yaw:
            yaw = 0
        next_state[6:10] = ang_to_quat([roll, pitch, yaw])

        return next_state

    
    def extract_mean_and_cov_from_state_sigma_points(self, sps):
        mu_bar = self.w0_m * sps[:, 0] + self.wi*np.sum(sps[:, 1:], 1)
        mu_bar[6:10], ei_vec_set = average_quaternions(sps[6:10, :].T, self.w_arr_mean)

        Wprime = np.zeros((self.dim_sig, sps.shape[1]))
        sig_bar = np.zeros((self.dim_sig, self.dim_sig))
        for sp_ind in range(sps.shape[1]):
            Wprime[0:6, sp_ind] = sps[0:6, sp_ind] - mu_bar[0:6]  # still need to overwrite the quat parts of this
            Wprime[9:12, sp_ind] = sps[10:13, sp_ind] - mu_bar[10:13]  # still need to overwrite the quat parts of this
            Wprime[6:9, sp_ind] = ei_vec_set[:, sp_ind];
            sig_bar = sig_bar + self.w_arr_cov[sp_ind] * np.matmul(Wprime[:, sp_ind], Wprime[:, sp_ind].T) ### TEMP PYTHON 2 ###
            # sig_bar = sig_bar + self.w_arr_cov[sp_ind] * Wprime[:, sp_ind] @ Wprime[:, sp_ind].T
        
        sig_bar = sig_bar + self.Q  # add noise
        sig_bar = enforce_pos_def_sym_mat(sig_bar) # project sig_bar to pos. def. cone to avoid numeric issues
        return mu_bar, sig_bar
        

    def extract_mean_and_cov_from_obs_sigma_points(self, sps_meas):
        z_hat = sps_meas[:, 0]
        S = copy(sps_meas)
        S = enforce_pos_def_sym_mat(S) # project S to pos. def. cone to avoid numeric issues
        return z_hat, S


    def calculate_cross_correlation():
        pass


