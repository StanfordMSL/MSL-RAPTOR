
# IMPORTS
# system & tools
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import copy
import time
import pdb
import os
import yaml
# vision
import cv2
# math
import numpy as np
import numpy.linalg as la

import random
# libs & utils
from utils_msl_raptor.ukf_utils import *
from utils_msl_raptor.ros_utils import *
from utils_msl_raptor.math_utils import *


class UKF:

    def __init__(self, camera, bb_3d, obj_width, obj_height, ukf_prms, init_time=0.0, b_use_gt_bb=False, class_str='mslquad', obj_id=0,verbose=False):

        self.verbose = verbose

        # Paramters #############################
        self.dim_state = 13
        self.dim_sig = 12  # covariance is 1 less dimension due to quaternion
        self.dim_meas = 5  # angled bounding box: row, col, width, height, angle
        self.b_use_gt_bb = b_use_gt_bb
        self.camera = camera

        self.class_str = class_str  # class name (string) e.g. 'person' or 'mslquad'
        self.obj_id = obj_id  # a unique int classifier
        self.projected_3d_bb = None  # r, c of projection of the estimated 3d bounding box
        
        self.ukf_prms = ukf_prms
        
        self.b_enforce_0_yaw   = bool(self.ukf_prms['b_enforce_0_yaw'])
        self.b_enforce_z       = bool(self.ukf_prms['b_enforce_z'])
        self.b_enforce_0_pitch = bool(self.ukf_prms['b_enforce_0_pitch'])
        self.b_enforce_0_roll  = bool(self.ukf_prms['b_enforce_0_roll'])
        kappa = float(self.ukf_prms['kappa'])
        self.w0 = kappa / (kappa + self.dim_sig)
        self.wi = 1 / (2 * (kappa + self.dim_sig))
        self.w_arr = np.ones((1+ 2 * self.dim_sig,)) * self.wi
        self.w_arr[0] = self.w0
        self.sig_pnt_multiplier = np.sqrt(self.dim_sig + kappa)

        self.init_filter_elements()
        
        ####################################################################

        # init vars #############################
        self.bb_3d = bb_3d
        self.obj_width = obj_width
        self.obj_height = obj_height
        self.itr = 0
        self.itr_time_prev = init_time
        self.itr_time = init_time
        self.tf_ego_w_tmp = None
        


    def init_filter_elements(self, mu=None):
        self.last_dt = 0.03
        if self.ukf_prms is not None:
            # if True:  # this is for DEBUGGING (its easier to try different values)
            self.sigma = np.diag([float(self.ukf_prms['dp_sigma'][0]), float(self.ukf_prms['dp_sigma'][1]), float(self.ukf_prms['dp_sigma'][2]), \
                                  float(self.ukf_prms['dv_sigma'][0]), float(self.ukf_prms['dv_sigma'][1]), float(self.ukf_prms['dv_sigma'][2]), \
                                  float(self.ukf_prms['dq_sigma'][0]), float(self.ukf_prms['dq_sigma'][1]), float(self.ukf_prms['dq_sigma'][2]), \
                                  float(self.ukf_prms['dw_sigma'][0]), float(self.ukf_prms['dw_sigma'][1]), float(self.ukf_prms['dw_sigma'][2])])
            # self.Q = self.sigma / float(self.ukf_prms['Q_div_fact'])  # Process Noise
            self.Q = np.diag([float(self.ukf_prms['dp_q'][0]), float(self.ukf_prms['dp_q'][1]), float(self.ukf_prms['dp_q'][2]), \
                              float(self.ukf_prms['dv_q'][0]), float(self.ukf_prms['dv_q'][1]), float(self.ukf_prms['dv_q'][2]), \
                              float(self.ukf_prms['dq_q'][0]), float(self.ukf_prms['dq_q'][1]), float(self.ukf_prms['dq_q'][2]), \
                              float(self.ukf_prms['dw_q'][0]), float(self.ukf_prms['dw_q'][1]), float(self.ukf_prms['dw_q'][2])])
            self.R = np.diag(self.ukf_prms['R'])  # Measurement Noise
            # else:
            #     self.sigma = np.asarray(self.ukf_prms['simga0'])
            #     self.Q = self.sigma/float(self.ukf_prms['Q_div_fact'])  # Process Noise
            #     self.R = np.diag(self.ukf_prms['R'])  # Measurement Noise
        else:
            dp = 0.1  # [m]
            dv = 0.005  # [m/s]
            dq = 0.1  # [rad] in ax ang 
            dw = 0.005  # [rad/s]
            self.sigma = np.diag([dp, dp, dp, dv, dv, dv, dq, dq, dq, dw, dw, dw])

            self.Q = self.sigma/10  # Process Noise
            self.R = np.diag([2, 2, 10, 10, 0.08])  # Measurement Noise
        
        if mu is None:
            self.mu = np.zeros((self.dim_state, ))  # set by main function initialization
        else:
            self.mu = mu


    def step_ukf(self, measurement, tf_ego_w, itr_time):
        """
        UKF iteration following pseudo code from probablistic robotics
        """
        # Calculate dt based on current and previous iteration times
        self.itr_time = itr_time
        if self.itr_time == self.itr_time_prev: # first run through
            dt = self.last_dt
        else:
            dt = self.itr_time - self.itr_time_prev
        
        # line 2
        # Rescale noises based on dt
        self.sigma = enforce_pos_def_sym_mat(self.sigma*(dt/self.last_dt))
        self.Q = self.Q*(dt/self.last_dt)
        self.R = self.R*(dt/self.last_dt)

        self.last_dt = dt  # store previous dt

        b_outer_only = True
        tic0 = time.time()
        sps = self.calc_sigma_points(self.mu, self.sigma)
        if not b_outer_only:
            print("calc sig pnts1: {:.4f}".format(time.time() - tic0))

        # line 3
        if not b_outer_only:
            tic = time.time()
        sps_prop = self.propagate_dynamics(sps, dt)
        if not b_outer_only:
            print("propagate_dynamics: {:.4f}".format(time.time() - tic))

        # lines 4 & 5
        if not b_outer_only:
            tic = time.time()
        mu_bar, sig_bar = self.extract_mean_and_cov_from_state_sigma_points(sps_prop)
        if not b_outer_only:
            print("extract_mean_and_cov_from_STATE_sigma_points: {:.4f}".format(time.time() - tic))
        
        # line 6
        if not b_outer_only:
            tic = time.time()
        sps_recalc = self.calc_sigma_points(mu_bar, sig_bar)
        if not b_outer_only:
            print("calc sig pnts2: {:.4f}".format(time.time() - tic))
        
        # lines 7-9
        if not b_outer_only:
            tic = time.time()
        pred_meas = np.zeros((self.dim_meas, sps.shape[1]))
        for sp_ind in range(sps.shape[1]):
            pred_meas[:, sp_ind] = self.predict_measurement(sps_recalc[:, sp_ind], tf_ego_w, measurement=measurement)
        if not b_outer_only:
            print("pred_meas: {:.4f}".format(time.time() - tic))

        if not b_outer_only:
            tic = time.time()
        z_hat, S, S_inv = self.extract_mean_and_cov_from_obs_sigma_points(pred_meas)
        self.mu_obs = z_hat
        self.S_obs = S
        if not b_outer_only:
            print("extract_mean_and_cov_from_OBS_sigma_points: {:.4f}".format(time.time() - tic))

        # line 10
        if not b_outer_only:
            tic = time.time()
        S_xz = self.calc_cross_correlation(sps_recalc, mu_bar, z_hat, pred_meas)
        if not b_outer_only:
            print("calc_cross_correlation: {:.4f}".format(time.time() - tic))

        # lines 11-13
        if not b_outer_only:
            tic = time.time()
        mu_out, sigma_out = self.update_state(measurement, mu_bar, sig_bar, S, S_inv, S_xz, z_hat)

        self.mu = mu_out
        self.sigma = sigma_out

        self.itr += 1
        self.itr_time_prev = self.itr_time
        tic1 = time.time()
        if self.verbose:
            if not b_outer_only:
                print("update_state: {:.4f}\nTOTAL time (with prints): {:.4f}".format(tic1 - tic, tic1 - tic0))
            else:
                print("TOTAL time (no prints): {:.4f}".format(tic1 - tic0))


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


        # print("mu: {}".format(self.mu))
        # print("mu_out: {}".format(mu_out))
        # print("z: {}   z_hat = {}".format(z, z_hat))
        # print("z - z_hat: {}".format(z - z_hat))
        # print("K: {}".format(k))
        # print("innovation: {}".format(innovation))

        if self.b_enforce_0_roll:
            mu_out[6:10] = remove_roll(mu_out[6:10])
        if self.b_enforce_0_pitch:
            mu_out[6:10] = remove_pitch(mu_out[6:10])
        if self.b_enforce_0_yaw:
            mu_out[6:10] = remove_yaw(mu_out[6:10])
        if self.b_enforce_z:
            mu_out[2] = self.bb_3d[0,2]


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
        S = self.w_arr*(sps_meas - z_hat_2d) @ (sps_meas - z_hat_2d).T

        S += self.R  # add measurement noise
        S = enforce_pos_def_sym_mat(S) # project S to pos. def. cone to avoid numeric issues
        S_inv = la.inv(S)
        return z_hat, S, S_inv


    def predict_measurement(self, state, tf_ego_w, measurement=None):
        """
        OUTPUT: (col[x], row[y], width, height, angle[RADIANS])
        use camera model & relative states to predict the angled 2d 
        bounding box seen by the ego drone 
        Pass in actual measurement to help resolve 90deg box rotation ambiguities 
        """

        # get relative transform from camera to quad
        tf_w_ado = state_to_tf(state)
        tf_cam_ado = self.camera.tf_cam_ego @ tf_ego_w @ tf_w_ado

        # tranform 3d bounding box from quad frame to camera frame
        bb_3d_cam = (tf_cam_ado @ self.bb_3d.T).T[:, 0:3]

        bb_rc_list = pose_to_3d_bb_proj(tf_w_ado, inv_tf(tf_ego_w), self.bb_3d, self.camera)
        # construct sensor output
        # minAreaRect sometimes flips the w/h and angle from how we want the output to be
        # to fix this, we can use boxPoints to get the x,y of the bb rect, and use our function
        # to get the output in the form we want 
        rect = cv2.minAreaRect(np.fliplr(bb_rc_list.astype('float32')))  # apparently float64s cause this function to fail
        box = cv2.boxPoints(rect)
        output = bb_corners_to_angled_bb(box, output_coord_type='xy')
        if measurement is not None:
            ang_thesh = np.deg2rad(20)  # angle threshold for considering alternative box rotation
            alt_ang = -np.sign(output[-1]) * (np.pi/2 - np.abs(output[-1]))  # negative complement of angle
            
            if (abs(alt_ang - measurement[-1]) < abs(output[-1] - measurement[-1])) and np.abs((np.abs(measurement[-1] - output[-1]) - np.pi/2)) < ang_thesh:
                output[-1] = alt_ang 
                w = output[2]
                h = output[3]
                output[2] = h
                output[3] = w
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
        if self.b_enforce_z:
            sig_step_all[2, :] = self.bb_3d[0,2]

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

        # General point mass model

        if self.class_str.lower() == 'person':
            # People on on the ground
            # update position
            # Get unique vector pointing towards heading. Maybe use heading from next_states?
            yaw_angs = quat_to_ang(states[6:10,:].T)[:,2]
            heading_vecs = np.array([np.cos(yaw_angs),np.sin(yaw_angs)])
            
            next_states[0:2,:] += dt  * np.sum(np.multiply(heading_vecs,states[3:5,:]),axis=0) * heading_vecs # no z update, projected on heading vector

            # update orientation
            quat = states[6:10,:].T  # current orientation
            omegas = states[10:13,:]  # angular velocity vector
            om_norm = la.norm(omegas,axis=0)  # rate of change of all angles
            om_norm[np.argwhere(om_norm == 0)] = 1
            ang = om_norm * dt  # change in angle in this small timestep
            ax = omegas / om_norm  # axis about angle change
            quat_delta = axang_to_quat((ax * ang).T)
            quat_new = quat_mul(quat_delta, quat)

            next_states[6:10,:] = quat_new.T

        else :

            # update position
            next_states[0:3,:] += dt * states[3:6,:]

            # update orientation
            quat = states[6:10,:].T  # current orientation
            omegas = states[10:13,:]  # angular velocity vector
            om_norm = la.norm(omegas,axis=0)  # rate of change of all angles
            om_norm[np.argwhere(om_norm == 0)] = 1
            ang = om_norm * dt  # change in angle in this small timestep
            ax = omegas / om_norm  # axis about angle change
            quat_delta = axang_to_quat((ax * ang).T)
            quat_new = quat_mul(quat_delta, quat)

            next_states[6:10,:] = quat_new.T
            
    
            
        if self.b_enforce_0_roll:
            next_states[6:10,:] = remove_roll(quat_new).T
        if self.b_enforce_0_pitch:
            next_states[6:10,:] = remove_pitch(quat_new).T
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


    def approx_pose_from_bb(self,bb,tf_w_ego):
        """
        Initialize a state with approximations using a single bounding box
        """
        if self.obj_width < self.obj_height:
            width = min(bb[2],bb[3])
            height = max(bb[2],bb[3])
        else :
            width = max(bb[2],bb[3])
            height = min(bb[2],bb[3])

        z = self.camera.new_camera_matrix[0,0]* self.obj_width /width
        im_coor = z*np.array([bb[0],bb[1],1.0])
        pos = self.camera.new_camera_matrix_inv @ im_coor
        pos = tf_w_ego @ inv_tf(self.camera.tf_cam_ego) @ np.concatenate([pos, [1]])
        # Only roll from the angle of the box
        quat = ang_to_quat(np.array([[bb[-1],0,0]])).flatten()
        # quat = np.array([1.,0.,0.,0.])
        return pos[:3],quat


    def reinit_filter_approx(self,bb,tf_w_ego):
        """
        Initialize a state with approximations using a single bounding box
        """
        pos,quat = self.approx_pose_from_bb(bb,tf_w_ego)
        print(pos)
        print(quat)
        mu = np.array([pos[0],pos[1],pos[2],0.,0.,0.,quat[0],quat[1],quat[2],quat[3],0.,0.,0.])
        self.init_filter_elements(mu)


    def reinit_filter_from_gt(self,pose):
        """
        Initialize a state with groundtruth pose
        """
        print('USING GROUND TRUTH TO INITIALIZE POSE')
        mu = np.array([pose[0],pose[1],pose[2],0.,0.,0.,pose[3],pose[4],pose[5],pose[6],0.,0.,0.])
        # Add noise
        noise = np.array([random.uniform(-0.02, 0.02) for i in range(3)]) 
        mu[:3] = mu[:3]+noise
        self.init_filter_elements(mu)


