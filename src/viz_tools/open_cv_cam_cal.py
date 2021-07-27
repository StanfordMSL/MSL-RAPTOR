#!/usr/bin/env python3
# IMPORTS
# system
import sys, os, glob
from copy import copy
import pdb
# math
import numpy as np
from bisect import bisect_left
# ros
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import tf
# libs & utils
sys.path.insert(1, '/root/msl_raptor_ws/src/msl_raptor/src')
from utils_msl_raptor.ros_utils import pose_to_tf, get_ros_time
from utils_msl_raptor.ukf_utils import inv_tf
from utils_msl_raptor.viz_utils import draw_2d_proj_of_3D_bounding_box
import cv2
from cv_bridge import CvBridge, CvBridgeError

class calibrate_camera:
    """
    This rosnode has two modes. In mode 1 it publishes a white background with the bounding boxes drawn (green when tracking, red when detecting). 
    This is faster and less likely to slow down the network. Mode 2 publishes the actual image. This is good for debugging, but is slower.
    If rosparam b_overlay is false (default), it will be mode 1, else mode 2.
    """

    def __init__(self):

        path_to_imgs = "/mounted_folder/rosbags_for_post_process/camera_cal/2021_07_26/chosen_images/"

        # Define the dimensions of checkerboard
        CHECKERBOARD = (8, 13)
        # CHECKERBOARD = (9, 14)
        
        # stop the iteration when specified
        # accuracy, epsilon, is reached or
        # specified number of iterations are completed.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Vector for 3D points
        threedpoints = []
        
        # Vector for 2D points
        twodpoints = []
        
        
        #  3D points real world coordinates
        objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None

        images = glob.glob(path_to_imgs + "*.jpg")
        image = None
        for filename in images:
            image = cv2.imread(filename)
            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
            # Find the chess board corners
            # If desired number of corners are
            # found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(
                                grayColor, CHECKERBOARD,
                                cv2.CALIB_CB_ADAPTIVE_THRESH
                                + cv2.CALIB_CB_FAST_CHECK +
                                cv2.CALIB_CB_NORMALIZE_IMAGE)
        
            # If desired number of corners can be detected then,
            # refine the pixel coordinates and display
            # them on the images of checker board
            if ret == True:
                threedpoints.append(objectp3d)
        
                # Refining pixel coordinates
                # for given 2d points.
                corners2 = cv2.cornerSubPix( grayColor, corners, (11, 11), (-1, -1), criteria)
        
                twodpoints.append(corners2)
        
                # Draw and display the corners
                image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
                

 
            cv2.imshow('img', image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

        pdb.set_trace()
    
        h, w = image.shape[:2]

        # Perform camera calibration by
        # passing the value of above found out 3D points (threedpoints)
        # and its corresponding pixel coordinates of the
        # detected corners (twodpoints)
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
            threedpoints, twodpoints, grayColor.shape[::-1], None, None)
        
        
        # Displaying required output
        print(" Camera matrix:")
        print(matrix)
        
        print("\n Distortion coefficient:")
        print(distortion)
        
        # print("\n Rotation Vectors:")
        # print(r_vecs)
        
        # print("\n Translation Vectors:")
        # print(t_vecs)

     


if __name__ == '__main__':
    np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
    try:
        program = calibrate_camera()
        program.run()
    except:
        import traceback
        traceback.print_exc()
    print("--------------- FINISHED ---------------")

