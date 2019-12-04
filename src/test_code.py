#!/usr/bin/env python3

# IMPORTS
# system
import os, sys, argparse, time
import pdb
# from pathlib import Path
# save/load
# import pickle
# math
import numpy as np
# plots
# import matplotlib
# from matplotlib import pyplot as plt
# from mpl_toolkits import mplot3d
# ros
import rospy
# custom modules
from ros_interface import ros_interface as ROS
from ukf import UKF
# libs & utils
from utils.ros_utils import *
from utils.math_utils import *


class camera:
    def __init__(self):
        # camera intrinsic matrix K and pose relative to the quad (fixed)
        self.K = np.array([[483.50426183,   0.  , 318.29104565], [  0. , 483.89448247, 248.02496288], [  0. ,   0. ,   1.     ]])  # MY CAMERA
        self.K = np.array([[617.2744 ,    0,  324.1011], [0 , 617.3357,  241.5791], [0 ,        0  ,  1.0000]])  # MATLAB's CAMERA
        self.tf_cam_ego = np.array([[ 0.  ,  0.  ,  1.  ,  0.05],
                                 [-1.  ,  0.  ,  0.  ,  0.  ],
                                 [ 0.  , -1.  ,  0.  ,  0.07],
                                 [ 0.  ,  0.  ,  0.  ,  1.  ]])

    def pnt3d_to_pix(self, pnt_c):
        """ input: assumes pnt in camera frame output: [row, col] i.e. the projection of xyz onto camera plane """
        rc = self.K @ np.reshape(pnt_c[0:3], 3, 1)
        return np.array([rc[1], rc[0]]) / rc[2]


if __name__ == '__main__':
    ukf = UKF()
    ukf.mu = np.array([-0.88312477, -0.15021993,  0.83246046,  0.,  0.  ,  0.  ,  0.99925363, -0.00460481,  0.00205432,  0.03829992,  0.   , 0 ,  0.])
    ukf.camera = camera()
    ukf.bb_3d = np.array([[ 0.135,  0.135,  0.065,  1.], [ 0.135,  0.135, -0.065,  1.   ], [ 0.135, -0.135, -0.065,  1.   ], [ 0.135, -0.135,  0.065,  1.   ], [-0.135, -0.135,  0.065,  1.   ], [-0.135, -0.135, -0.065,  1.   ], [-0.135,  0.135, -0.065,  1.   ], [-0.135,  0.135,  0.065,  1.   ]])

    tf_ego_w = np.array([[ 0.99903845,  0.01473524,  0.04129215,  3.6821852 ],
                      [-0.01510534,  0.99984836,  0.00866519,  0.14882801],
                      [-0.0411582 , -0.00928059,  0.99910954, -0.67069694],
                      [ 0.        ,  0.        ,  0.        ,  1.        ]])
    dt = 0.9344708919525146  # 0.0334
    measurement = ukf.predict_measurement(ukf.mu, tf_ego_w)
    for i in range(10):
        ukf.step_ukf(measurement, tf_ego_w, dt)
        ukf.itr_time += dt

    print('')
