
# IMPORTS
# system & tools
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import copy
import cv2
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
from utils_msl_raptor.ukf_utils import *
from utils_msl_raptor.ros_utils import *
from utils_msl_raptor.math_utils import *


class UKF:

    def __init__(self,b_enforce_0_yaw=True,b_use_gt_bb=False):

        self.VERBOSE = True

        # Paramters #############################
        self.b_enforce_0_yaw = b_enforce_0_yaw
        self.b_use_gt_bb = b_use_gt_bb
        self.dim_state = 13
        self.dim_sig = 12  # covariance is 1 less dimension due to quaternion
        self.dim_meas = 5  # angled bounding box: row, col, width, height, angle

        kappa = 2  # based on State Estimation for Robotics (Barfoot)
        self.sig_pnt_multiplier = np.sqrt(self.dim_sig + kappa)

        self.w0 = kappa / (kappa + self.dim_sig)
        self.wi = 1 / (2 * (kappa + self.dim_sig))
        self.w_arr = np.ones((1+ 2 * self.dim_sig,)) * self.wi
        self.w_arr[0] = self.w0

        self.camera = None

        dp = 0.1  # [m]
        dv = 0.005  # [m/s]
        dq = 0.1  # [rad] in ax ang 
        dw = 0.005  # [rad/s]
        self.sigma = np.diag([dp, dp, dp, dv, dv, dv, dq, dq, dq, dw, dw, dw])

        self.Q = self.sigma/10  # Process Noise
        self.R = np.diag([2, 2, 10, 10, 0.08])  # Measurement Noise
        self.parallelize = True
        ####################################################################

        # init vars #############################
        self.mu = np.zeros((self.dim_state, 1))  # set by main function initialization
        self.bb_3d = np.zeros((8, 3))  # set by main function initialization
        self.itr = 0
        self.itr_time = 0
        self.tf_ego_w_tmp = None
        ####################################################################


    def step_ukf(self, measurement, tf_ego_w, dt):
        """
        UKF iteration following pseudo code from probablistic robotics
        """
        if b_vs_debug():
            print("Starting UKF Iteration {} (time: {:.3f}s)".format(self.itr, self.itr_time))
        else:
            rospy.loginfo("Starting UKF Iteration {} (time: {:.3f}s)".format(self.itr, self.itr_time))
        print(self.mu)
        
        # line 2
        sps = self.calc_sigma_points(self.mu, self.sigma)

        # line 3
        sps_prop = self.propagate_dynamics(sps, dt)
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
            if not b_vs_debug():
                pdb.set_trace()
        
        # lines 7-9
        if self.parallelize:
            my_pool = Pool(cpu_count() - 1)  # keep 1 core free for now to avoid lockup issues? [THIS IS PURELY SPECULATIVE ON MY PART!!!]
            b_use_partial = True
            if b_use_partial:
                # the "right" way (I think) - check time again alteran
                par_func = partial(self.predict_measurement_par, tf_ego_w)  # creates wrapper for function predict_meas with only one parameters (the state)
                result = my_pool.imap(func=par_func, iterable=sps_recalc.T)
            else:
                # w/o using partial (hacky way to pass in tf_ego_w):
                self.tf_ego_w_tmp = tf_ego_w
                result = my_pool.imap(func=self.predict_measurement_par2, iterable=sps_recalc.T)
            pred_meas = np.array(list(result), dtype=np.float32).T
            my_pool.close()
        else:
            pred_meas = np.zeros((self.dim_meas, sps.shape[1]))
            for sp_ind in range(sps.shape[1]):
                pred_meas[:, sp_ind] = self.predict_measurement(sps_recalc[:, sp_ind], tf_ego_w)
        z_hat, S, S_inv = self.extract_mean_and_cov_from_obs_sigma_points(pred_meas)

        # line 10
        S_xz = self.calc_cross_correlation(sps_recalc, mu_bar, z_hat, pred_meas)

        # lines 11-13
        mu_out, sigma_out = self.update_state(measurement, mu_bar, sig_bar, S, S_inv, S_xz, z_hat)

        self.mu = mu_out
        self.sigma = sigma_out
        self.itr += 1


    def update_state(self, z, mu_bar, sig_bar, S, S_inv, S_xz, z_hat):
        k = S_xz @ S_inv
        innovation = k @ (z - z_hat)
        if np.any(np.abs(innovation) > 2):
            print("stop here")
        mu_out = copy(mu_bar)
        sigma_out = copy(sig_bar)
        mu_out[0:6] += innovation[0:6]
        mu_out[6:10] = enforce_quat_format(quat_mul(axang_to_quat(innovation[6:9]), mu_bar[6:10]))
        mu_out[10:13] += innovation[9:12]

        if self.b_enforce_0_yaw:
            mu_out[6:10] = remove_yaw(mu_out[6:10])

        sigma_out -=  k @ S @ k.T
        sigma_out = enforce_pos_def_sym_mat(sigma_out) # project sigma_out to pos. def. cone to avoid numeric issues

        return mu_out, sigma_out


    def calc_cross_correlation(self, sps, mu_bar, z_hat, pred_meas):
        dim_covar = self.sigma.shape[0]
        num_sps = sps.shape[1]
        
        quat_ave_inv = quat_inv(mu_bar[6:10])
        Wprime = np.zeros((dim_covar, num_sps))

        idx_mu_not_q = list(np.arange(6)) + list(np.arange(10,13))
        idx_sigma_not_q = list(np.arange(6)) + list(np.arange(9,12)) 

        Wprime[idx_sigma_not_q, :]= sps[idx_mu_not_q, :] - mu_bar[idx_mu_not_q].reshape(-1,1)
        q = sps[6:10, :]
        q_diff = quat_mul(q.T, quat_ave_inv)
        axang_diff = quat_to_axang(q_diff)
        Wprime[6:9, :] = axang_diff.T
            
        z_hat_2d = np.expand_dims(z_hat, axis=1)

        w_arr = np.ones(num_sps)*self.wi
        w_arr[0] = self.w0
        sigma_xz = w_arr * Wprime @ (pred_meas - z_hat_2d).T

        return sigma_xz
        
    
    def extract_mean_and_cov_from_obs_sigma_points(self, sps_meas):
        # calculate mean
        z_hat = self.w0 * sps_meas[:, 0] + self.wi * np.sum(sps_meas[:, 1:], axis=1)

        # calculate covariance
        z_hat_2d = np.expand_dims(z_hat, axis=1)
        S = self.w0 * (sps_meas[:, 0:1] - z_hat_2d) @ (sps_meas[:, 0:1] - z_hat_2d).T
        dim_covar = self.dim_sig
        w_arr = np.ones(sps_meas.shape[1])*self.wi
        w_arr[0] = self.w0
        S = w_arr*(sps_meas - z_hat_2d) @ (sps_meas - z_hat_2d).T

        
        S += self.R  # add measurement noise
        S = enforce_pos_def_sym_mat(S) # project S to pos. def. cone to avoid numeric issues
        S_inv = la.inv(S)
        return z_hat, S, S_inv


    def predict_measurement_par2(self, state):
        """
        use camera model & relative states to predict the angled 2d 
        bounding box seen by the ego drone 
        """
        tf_ego_w = self.tf_ego_w_tmp
        # get relative transform from camera to quad
        tf_w_quad = state_to_tf(state)
        tf_cam_quad = self.camera.tf_cam_ego @ tf_ego_w @ tf_w_quad
        # tf_cam_quad = np.array([[-0.0956 ,  -0.9951  ,  0.0258 ,  -0.1867],
        #                          [   0.0593,   -0.0316,   -0.9977,    0.6696],
        #                          [   0.9936 ,  -0.0939 ,   0.0620,    4.7865],
        #                          [       0  ,       0 ,        0 ,   1.0000]])

        # tranform 3d bounding box from quad frame to camera frame
        bb_3d_cam = (tf_cam_quad @ self.bb_3d.T).T[:, 0:3]

        # transform each 3d point to a 2d pixel (row, col)
        bb_rc_list = np.zeros((bb_3d_cam.shape[0], 2))
        for i, bb_vert in enumerate(bb_3d_cam):
            bb_rc_list[i, :] = self.camera.pnt3d_to_pix(bb_vert)

        # construct sensor output
        # minAreaRect sometimes flips the w/h and angle from how we want the output to be
        # to fix this, we can use boxPoints to get the x,y of the bb rect, and use our function
        # to get the output in the form we want 
        rect = cv2.minAreaRect(np.fliplr(bb_rc_list.astype('float32')))  # apparently float64s cause this function to fail
        box = cv2.boxPoints(rect)
        output = bb_corners_to_angled_bb(box, output_coord_type='rc')
        return output


    def predict_measurement_par(self, tf_ego_w, state):
        """
        use camera model & relative states to predict the angled 2d 
        bounding box seen by the ego drone 
        """
        # get relative transform from camera to quad
        tf_w_quad = state_to_tf(state)
        tf_cam_quad = self.camera.tf_cam_ego @ tf_ego_w @ tf_w_quad
        # tf_cam_quad = np.array([[-0.0956 ,  -0.9951  ,  0.0258 ,  -0.1867],
        #                          [   0.0593,   -0.0316,   -0.9977,    0.6696],
        #                          [   0.9936 ,  -0.0939 ,   0.0620,    4.7865],
        #                          [       0  ,       0 ,        0 ,   1.0000]])

        # tranform 3d bounding box from quad frame to camera frame
        bb_3d_cam = (tf_cam_quad @ self.bb_3d.T).T[:, 0:3]

        # transform each 3d point to a 2d pixel (row, col)
        bb_rc_list = np.zeros((bb_3d_cam.shape[0], 2))
        for i, bb_vert in enumerate(bb_3d_cam):
            bb_rc_list[i, :] = self.camera.pnt3d_to_pix(bb_vert)

        # construct sensor output
        # minAreaRect sometimes flips the w/h and angle from how we want the output to be
        # to fix this, we can use boxPoints to get the x,y of the bb rect, and use our function
        # to get the output in the form we want 
        rect = cv2.minAreaRect(np.fliplr(bb_rc_list.astype('float32')))  # apparently float64s cause this function to fail
        box = cv2.boxPoints(rect)
        output = bb_corners_to_angled_bb(box, output_coord_type='rc')

        return output


    def predict_measurement(self, state, tf_ego_w):
        """
        use camera model & relative states to predict the angled 2d 
        bounding box seen by the ego drone 
        """

        # get relative transform from camera to quad
        tf_w_quad = state_to_tf(state)
        tf_cam_quad = self.camera.tf_cam_ego @ tf_ego_w @ tf_w_quad
        # tf_cam_quad = np.array([[-0.0956 ,  -0.9951  ,  0.0258 ,  -0.1867],
        #                          [   0.0593,   -0.0316,   -0.9977,    0.6696],
        #                          [   0.9936 ,  -0.0939 ,   0.0620,    4.7865],
        #                          [       0  ,       0 ,        0 ,   1.0000]])

        # tranform 3d bounding box from quad frame to camera frame
        bb_3d_cam = (tf_cam_quad @ self.bb_3d.T).T[:, 0:3]

        # transform each 3d point to a 2d pixel (row, col)
        bb_rc_list = np.zeros((bb_3d_cam.shape[0], 2))
        for i, bb_vert in enumerate(bb_3d_cam):
            bb_rc_list[i, :] = self.camera.pnt3d_to_pix(bb_vert)

        # construct sensor output
        # minAreaRect sometimes flips the w/h and angle from how we want the output to be
        # to fix this, we can use boxPoints to get the x,y of the bb rect, and use our function
        # to get the output in the form we want 
        rect = cv2.minAreaRect(np.fliplr(bb_rc_list.astype('float32')))  # apparently float64s cause this function to fail
        box = cv2.boxPoints(rect)
        output = bb_corners_to_angled_bb(box, output_coord_type='rc')
        return output


    def calc_sigma_points(self, mu, sigma):
        sps = np.zeros((self.dim_state, 2 * self.dim_sig + 1))
        sps[:, 0] = mu
        sig_step_p = self.sig_pnt_multiplier * la.cholesky(sigma).T
        sig_step_m = -sig_step_p
        sig_step_all = np.stack((sig_step_p,sig_step_m),2).reshape(self.dim_sig,-1)
        idx_mu_not_q = list(np.arange(6)) + list(np.arange(10,13))
        idx_sigma_not_q = list(np.arange(6)) + list(np.arange(9,12)) 


        sps[idx_mu_not_q,1:] = mu[idx_mu_not_q].reshape(-1,1) + sig_step_all[idx_sigma_not_q,:]
        if self.b_enforce_0_yaw:
            sig_step_all[8, :] = 0

        q_nom = mu[6:10]
        q_perturb = axang_to_quat(sig_step_all[6:9, :].T)
        sps[6:10, 1:] = quat_mul(q_perturb, q_nom).T
        return sps
    
    def propagate_dynamics(self, states, dt):
        """
        Estimate the next state vector. Assumes no control input (velocities stay the same)
        states are 13xn
        """
        states = states.reshape(13,-1)
        next_states = copy(states)

        # update position
        next_states[0:3,:] += dt * states[3:6,:]

        # update orientation
        quat = states[6:10,:].T  # current orientation
        omegas = states[10:13,:]  # angular velocity vector
        om_norm = la.norm(omegas,axis=0)  # rate of change of all angles
        # zeros_rot_idx = 
        # omega[:,zeros_rot_idx] = 0
        om_norm[np.argwhere(om_norm == 0)] = 1
        ang = om_norm * dt  # change in angle in this small timestep
        ax = omegas / om_norm  # axis about angle change
        quat_delta = axang_to_quat((ax * ang).T)
        quat_new = quat_mul(quat_delta, quat)

        next_states[6:10,:] = quat_new.T
        if self.b_enforce_0_yaw:
            next_states[6:10,:] = remove_yaw(quat_new).T

        return next_states
    
    def extract_mean_and_cov_from_state_sigma_points(self, sps):
        mu_bar = self.w0 * sps[:, 0] + self.wi*np.sum(sps[:, 1:], 1)
        mu_bar[6:10], ei_vec_set = average_quaternions(sps[6:10, :].T, self.w_arr)

        idx_mu_not_q = list(np.arange(6)) + list(np.arange(10,13))
        idx_sigma_not_q = list(np.arange(6)) + list(np.arange(9,12)) 

        Wprime = np.zeros((self.dim_sig, sps.shape[1]))
        sig_bar = np.zeros((self.dim_sig, self.dim_sig))

        Wprime[idx_sigma_not_q, :] = sps[idx_mu_not_q, :] - mu_bar[idx_mu_not_q].reshape(-1,1)  # still need to overwrite the quat parts of this
        Wprime[6:9, :] = ei_vec_set

        sig_bar = (self.w_arr * Wprime) @ Wprime.T
        
        sig_bar = enforce_pos_def_sym_mat(sig_bar + self.Q)  # add noise & project sig_bar to pos. def. cone to avoid numeric issues
        return mu_bar, sig_bar
