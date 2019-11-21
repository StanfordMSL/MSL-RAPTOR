
# IMPORTS
# math
import numpy as np
import numpy.linalg as la
# plots
# import matplotlib
# from matplotlib import pyplot as plt
# from mpl_toolkits import mplot3d
# ros
import rospy
# libs

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
        self.w0_c = self.w0_m + (1 - alpha**22 + beta)
        self.wi = 1 / (2*  (ukf_lambda + self.dim_sig))

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


    def step_ukf(self, measurement, bb_3d, tf_ego_w):
        self.ukf_itr += 1

        sps = self.calc_sigma_points()

    
    def calc_sigma_points(self):
        sps = np.zeros((self.dim_state, 2 * self.dim_sig + 1))
        sps[:, 0] = self.mu
        sig_step = self.sig_pnt_multiplier * la.cholesky(self.sigma)

        if self.b_enforce_0_yaw:
            sig_step[8, :] = 0

        quat_nom = self.mu[6:10]
        # loop over half (-1) the num sigma points and update two at once
        for sp_ind in range(self.dim_sig):
            sps[0:6, 1 + 2 * sp_ind] = self.mu[0:6] + sig_step[0:6, sp_ind]
            sps[9:, 1 + 2 * sp_ind] = self.mu[9:13] + sig_step[8:12, sp_ind]
            q_perturb = self.axang_to_quat(sig_step[6:9, sp_ind])
        return sps
    
    
    def propogate_dynamics():
        pass

    
    def extract_mean_and_cov_from_state_sigma_points():
        pass
        

    def extract_mean_and_cov_from_obs_sigma_points():
        pass


    def calculate_cross_correlation():
        pass


    def predict_measurement():
        pass


#####  UTULS  ##########

    def axang_to_quat(self, axang):
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