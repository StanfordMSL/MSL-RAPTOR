
# IMPORTS
#tools
from copy import copy
import pdb
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


class UKF:

    def __init__(self):

        self.VERBOSE = True

        # Paramters #############################
        self.b_enforce_0_yaw = True;
        self.dim_state = 13
        self.dim_sig = 12  # covariance is 1 less dimension due to quaternion

        alpha = .1  # scaling param - how far sig. points are from mean
        kappa = 2  # scaling param - how far sig. points are from mean
        beta = 2  # optimal choice according to probablistic robotics (textbook)
        ukf_lambda = alpha**2 * (self.dim_sig + kappa) - self.dim_sig
        self.sig_pnt_multiplier = np.sqrt(self.dim_sig + ukf_lambda)

        self.w0_m = ukf_lambda / (ukf_lambda + self.dim_sig)
        self.w0_c = self.w0_m + (1 - alpha**2 + beta)
        self.wi = 1 / (2 * (ukf_lambda + self.dim_sig))
        self.w_arr = np.ones((1+ 2 * self.dim_sig,)) * self.wi
        self.w_arr[0] = self.w0_m

        self.camera = {}
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
        self.ukf_itr += 1
        sps = self.calc_sigma_points()
        sps_prop = np.empty_like(sps)
        for sp_ind in range(sps_prop.shape[1]):
            sps_prop[:, sp_ind] = self.propagate_dynamics(sps[:, sp_ind], dt)

        mu_bar, sig_bar = self.extract_mean_and_cov_from_state_sigma_points(sps_prop)
        pdb.set_trace()
    
    def calc_sigma_points(self):
        sps = np.zeros((self.dim_state, 2 * self.dim_sig + 1))
        sps[:, 0] = self.mu
        sig_step = self.sig_pnt_multiplier * la.cholesky(self.sigma)

        if self.b_enforce_0_yaw:
            sig_step[8, :] = 0

        q_nom = self.mu[6:10]
        # loop over half (-1) the num sigma points and update two at once
        for sp_ind in range(self.dim_sig):
            # first step in positive direction
            sp_col_1 = 1 + 2 * sp_ind  # starting in the second col, count by pairs
            sps[0:6, sp_col_1] = self.mu[0:6] + sig_step[0:6, sp_ind]
            sps[10:, sp_col_1] = self.mu[10:13] + sig_step[9:12, sp_ind]
            q_perturb = axang_to_quat(sig_step[6:9, sp_ind])
            sps[6:10, sp_col_1] = quat_mul(q_perturb, q_nom)

            # next step in positive direction
            sp_col_2 = sp_col_1 + 1
            sps[0:6, sp_col_2] = self.mu[0:6] - sig_step[0:6, sp_ind]
            sps[10:, sp_col_2] = self.mu[10:13] - sig_step[9:12, sp_ind]
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
        mu_bar[6:10] = average_quaternions(sps[6:10, :].T, self.w_arr)
        sig_bar = 0
        return mu_bar, sig_bar
        

    def extract_mean_and_cov_from_obs_sigma_points():
        pass


    def calculate_cross_correlation():
        pass


    def predict_measurement():
        pass
