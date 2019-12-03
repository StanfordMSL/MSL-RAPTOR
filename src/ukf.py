
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
        self.b_enforce_0_yaw = True
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
        print(self.mu)

        # line 2
        sps = self.calc_sigma_points(self.mu, self.sigma)

        # line 3
        sps_prop = np.empty_like(sps)
        for sp_ind in range(sps.shape[1]):
            sps_prop[:, sp_ind] = self.propagate_dynamics(sps[:, sp_ind], dt)

        # lines 4 & 5
        mu_bar, sig_bar = self.extract_mean_and_cov_from_state_sigma_points(sps_prop)
        
        # line 6
        try:
            sps_recalc = self.calc_sigma_points(mu_bar, sig_bar)
        except Exception as e:
            print(e)
            print(sig_bar)
            eig_vals, eig_vecs = la.eig(sig_bar)
            print(eig_vals)
            pdb.set_trace()
        
        # lines 7-9
        pred_meas = np.zeros((self.dim_meas, sps.shape[1]))
        for sp_ind in range(sps.shape[1]):
            pred_meas[:, sp_ind] = self.predict_measurement(sps_recalc[:, sp_ind], bb_3d, tf_ego_w)
        z_hat, S, S_inv = self.extract_mean_and_cov_from_obs_sigma_points(pred_meas)

        # line 10
        S_xz = self.calc_cross_correlation(sps_recalc, mu_bar, z_hat, pred_meas)

        # lines 11-13
        mu_out, sigma_out = self.update_state(measurement, mu_bar, sig_bar, S, S_inv, S_xz, z_hat)

        self.mu = mu_out
        self.sigma = sigma_out
        self.ukf_itr += 1


    def update_state(self, z, mu_bar, sig_bar, S, S_inv, S_xz, z_hat):
        k = S_xz @ S_inv
        innovation = k @ (z - z_hat)
        mu_out = copy(mu_bar)
        sigma_out = copy(sig_bar)
        mu_out[0:6] += innovation[0:6]
        mu_out[6:10] = quat_mul(axang_to_quat(innovation[6:9]), mu_bar[6:10])
        mu_out[10:13] += innovation[9:12]

        if self.b_enforce_0_yaw:
            mu_out[6:10] = remove_yaw(mu_out)

        sigma_out -=  k @ S @ k.T
        sigma_out = enforce_pos_def_sym_mat(sigma_out) # project sigma_out to pos. def. cone to avoid numeric issues

        return mu_out, sigma_out


    def calc_cross_correlation(self, sps, mu_bar, z_hat, pred_meas):
        dim_covar = self.sigma.shape[0]
        num_sps = sps.shape[1]
        
        quat_ave_inv = quat_inv(mu_bar[6:10])
        Wprime = np.zeros((dim_covar, num_sps))
        for sp_ind in range(num_sps):
            Wprime[0:6, sp_ind] = sps[:6, sp_ind] - mu_bar[0:6]
            Wprime[9:12, sp_ind] = sps[10:13, sp_ind] - mu_bar[10:13] 
            
            q = sps[6:10, sp_ind]
            q_diff = quat_mul(q, quat_ave_inv)
            axang_diff = quat_to_axang(q_diff)
            Wprime[6:9, sp_ind] = axang_diff

        z_hat_2d = np.expand_dims(z_hat, axis=1)
        # note: [:, 0:1] results in shape (n,1) vs. []:, 0] --> shape of (n,)
        sigma_xz = self.w0_c * Wprime[:, 0:1] @ (pred_meas[:, 0:1] - z_hat_2d).T
        for i in range(1, num_sps):
            sigma_xz = sigma_xz + self.wi * Wprime[:, i:i+1] @ (pred_meas[:, i:i+1] - z_hat_2d).T

        return sigma_xz
        
    
    def extract_mean_and_cov_from_obs_sigma_points(self, sps_meas):
        # calculate mean
        z_hat = self.w0_m * sps_meas[:, 0] + self.wi * np.sum(sps_meas[:, 1:], axis=1)

        # calculate covariance
        S = self.w0_c * (sps_meas[:, 0] - z_hat) @ (sps_meas[:, 0] - z_hat).T
        dim_covar = sps_meas.shape[0]
        for obs_ind in range(dim_covar):
            S = S + self.wi * (sps_meas[:, obs_ind + 2] - z_hat) @ (sps_meas[:, obs_ind + 2] - z_hat).T
            S = S + self.wi * (sps_meas[:, dim_covar + obs_ind + 2] - z_hat) @ (sps_meas[:, dim_covar + obs_ind + 2] - z_hat).T
        
        S += self.R  # add measurement noise
        S = enforce_pos_def_sym_mat(S) # project S to pos. def. cone to avoid numeric issues
        S_inv = la.inv(S)
        return z_hat, S, S_inv


    def predict_measurement(self, state, bb_3d_quad, tf_ego_w):
        """
        use camera model & relative states to predict the angled 2d 
        bounding box seen by the ego drone 
        """

        # get relative transform from camera to quad
        tf_w_quad = state_to_tf(state)
        
        tf_cam_quad = self.camera.tf_cam_ego @ tf_ego_w @ tf_w_quad
        
        # tranform 3d bounding box from quad frame to camera frame
        bb_3d_cam = (tf_cam_quad @ bb_3d_quad.T)[:, 0:3]

        # transform each 3d point to a 2d pixel (row, col)
        bb_cr_list = np.zeros((bb_3d_cam.shape[0], 2))
        for i, bb_vert in enumerate(bb_3d_cam):
            bb_cr_list[i, :] = self.camera.pnt3d_to_pix(bb_vert)

        # construct sensor output
        # minAreaRect sometimes flips the w/h and angle from how we want the output to be
        # to fix this, we can use boxPoints to get the x,y of the bb rect, and use our function
        # to get the output in the form we want 
        rect = cv2.minAreaRect(bb_cr_list.astype('float32'))  # apparently float64s cause this function to fail
        box = cv2.boxPoints(rect)
        output = bb_corners_to_angled_bb(box, output_coord_type='rc')
        # pdb.set_trace()
        return output


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

        next_state[6:10] = quat_new
        if self.b_enforce_0_yaw:
            next_state[6:10] = remove_yaw(quat_new)

        return next_state

    
    def extract_mean_and_cov_from_state_sigma_points(self, sps):
        mu_bar = self.w0_m * sps[:, 0] + self.wi*np.sum(sps[:, 1:], 1)
        mu_bar[6:10], ei_vec_set = average_quaternions(sps[6:10, :].T, self.w_arr_mean)

        Wprime = np.zeros((self.dim_sig, sps.shape[1]))
        sig_bar = np.zeros((self.dim_sig, self.dim_sig))
        for sp_ind in range(sps.shape[1]):
            Wprime[0:6, sp_ind] = sps[0:6, sp_ind] - mu_bar[0:6]  # still need to overwrite the quat parts of this
            Wprime[9:12, sp_ind] = sps[10:13, sp_ind] - mu_bar[10:13]  # still need to overwrite the quat parts of this
            Wprime[6:9, sp_ind] = ei_vec_set[:, sp_ind]
            sig_bar = sig_bar + self.w_arr_cov[sp_ind] * Wprime[:, sp_ind] @ Wprime[:, sp_ind].T
        
        sig_bar = enforce_pos_def_sym_mat(sig_bar + self.Q)  # add noise & project sig_bar to pos. def. cone to avoid numeric issues
        return mu_bar, sig_bar
