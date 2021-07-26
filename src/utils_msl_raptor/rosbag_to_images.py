#!/usr/bin/env python3
import sys, os, time
from copy import copy
import pdb

import numpy as np

import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
# sys.path.insert(1, '/root/msl_raptor_ws/src/msl_raptor/src/front_end/coral/tflite/python/examples/detection')
# # sys.path.append(os.path.dirname(os.path.dirname('/root/msl_raptor_ws/src/msl_raptor/src/front_end/coral/tflite/python/examples/detection')))
# import detect_image_coral
# import detect_coral

# sys.path.insert(1, '/root/msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask')
sys.path.insert(1, '/root/msl_raptor_ws/src/msl_raptor/src/front_end')
sys.path.insert(1, '/root/msl_raptor_ws/src/msl_raptor/src')
from tracker import SiammaskTracker
from utils_msl_raptor.ros_utils import *
from utils_msl_raptor.math_utils import *


import argparse
import time
from PIL import Image
from PIL import ImageDraw
import tflite_runtime.interpreter as tflite
import platform

from utils_msl_raptor.ukf_utils import bb_corners_to_angled_bb


class rosbag_to_imabges:

    def __init__(self):

        topic_str = ['/quad7/camera/image_raw'] # , '/vrpn_client_node/quad7/pose', '/vrpn_client_node/bowl_green_msl/pose']
        base_dir = '/mounted_folder/rosbags_for_post_process/camera_cal/2021_07_26/'
        img_save_dir = base_dir + 'extracted_images/'
        rosbag_name_and_path = base_dir + 'camera_cal_bag_2021_07_26_13.bag'
        if not os.path.exists(img_save_dir):
            os.mkdir(img_save_dir)
        
        # definitions for loop
        self.bridge = CvBridge()
        im_save_idx = 0
        bag_in = rosbag.Bag(rosbag_name_and_path, 'r') 
        for topic, msg, t in bag_in.read_messages(topics=topic_str):
            if topic == '/quad7/camera/image_raw':
                image_cv2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cv2.imwrite(img_save_dir + 'image_{:04d}'.format(im_save_idx) + '.jpg', image_cv2)
                im_save_idx += 1
                continue  
        bag_in.close()

        print('Done with bag!')


if __name__ == '__main__':
    np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
    try:
        rosbag_to_imabges()
    except:
        import traceback
        traceback.print_exc()
    print("--------------- FINISHED ---------------")
