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


import argparse
import time
from PIL import Image
from PIL import ImageDraw
import tflite_runtime.interpreter as tflite
import platform

from utils_msl_raptor.ukf_utils import bb_corners_to_angled_bb


class segment_object_pixels:
    def __init__(self):
        b_mask = True
        self.bridge = CvBridge()
        rb_path = '/mounted_folder/bags_to_test_coral_detect/'
        # rb_name = 'grey_bowl_msl/bowl_grey_msl_nerf_with_markers.bag'  
        rb_name = 'green_bowl_msl/bowl_green_msl_nerf_with_markers.bag'
        img_out_path = rb_path + rb_name[:-4] + '_output/'
        if not os.path.exists(img_out_path):
             os.mkdir(img_out_path)
        img_det_out_path = img_out_path + 'img_with_detection/'
        if not os.path.exists(img_det_out_path):
             os.mkdir(img_det_out_path)

        seg_im_out_path = img_out_path + 'segmentation_masks/'
        seg_np_out_path = seg_im_out_path + "seg_as_numpy/"
        if b_mask and not os.path.exists(seg_im_out_path):
            os.mkdir(seg_im_out_path)
        if b_mask and not os.path.exists(seg_np_out_path):
            os.mkdir(seg_np_out_path)


        bag_in = rosbag.Bag(rb_path + rb_name, 'r')
        base_dir = '/root/msl_raptor_ws/src/msl_raptor/src/front_end/'

        # label_file = '/mounted_folder/models/coco_labels.txt'
        # model_file = '/mounted_folder/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'

        # labels = detect_image_coral.load_labels(label_file) if label_file else {}
        # interpreter = detect_image_coral.make_interpreter(model_file)
        # interpreter.allocate_tensors()
        # scale = None
        # thresh = 0.4
        b_save_output = True
        b_first_loop = True
        ave_time = 0
        max_time = 0

        # topic_str = '/camera/image_raw'
        topic_str = '/quad7/camera/image_raw'
        im_idx = 0
        for topic, msg, t in bag_in.read_messages(topics=topic_str):
            image_cv2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            img_path_and_name = img_out_path + 'image_{:04d}'.format(im_idx) + '.jpg'
            
            if b_first_loop:
                b_first_loop = False
                self.tracker = SiammaskTracker(image_cv2, base_dir=base_dir, use_tensorrt=False)
                # # GREY BOWL: 285, 275  | 395, 275  | 395, 330  | 285, 330
                # init_bb = ((285 + 395)/2, (275+ 330)/2, 395 - 285, 330 - 275)  # box format: x,y,w,h (where x,y correspond to the top left corner)
                # init_bb_up_left = (285, 275, 395 - 285, 330 - 275)  # box format: x,y,w,h (where x,y correspond to the top left corner)
                # GREEN BOWL: 290, 255  | 335, 255  | 335, 277  | 290, 277
                init_bb = ((290 + 335)/2, (255+ 277)/2, 335 - 290, 277 - 255)  # box format: x,y,w,h (where x,y correspond to the top left corner)
                init_bb_up_left = (290, 255, 335 - 290, 277 - 255)  # box format: x,y,w,h (where x,y correspond to the top left corner)
                box0 = np.int0(cv2.boxPoints( ( (init_bb[0], init_bb[1]), (init_bb[2], init_bb[3]), 0)) )
                next_state = self.tracker.reinit(init_bb_up_left, image_cv2)
                redImg = np.zeros(image_cv2.shape, image_cv2.dtype)
                redImg[:,:] = (0, 0, 255)
                
            next_state, abb, mask = self.tracker.track(image_cv2, next_state)
            abb = bb_corners_to_angled_bb(abb.reshape(-1,2))

            if b_save_output:
                img_path_and_name_result = img_det_out_path + 'image_result_{:04d}'.format(im_idx) + '.jpg'
                box = np.int0(cv2.boxPoints( ( (abb[0], abb[1]), (abb[2], abb[3]), -np.degrees(abb[4]))) )
                image_cv2_modified = copy(image_cv2)
                if b_first_loop:
                    cv2.drawContours(image_cv2_modified, [box0], 0, (255,0,0), 2)  # draw init box
                cv2.drawContours(image_cv2_modified, [box], 0, (0,255,0), 2)
                
                if b_mask:
                    np_mask = np.array(mask, dtype=np.uint8)
                    redMask = cv2.bitwise_and(redImg, redImg, mask=np_mask)
                    alpha = 0.3
                    cv2.addWeighted(redMask, alpha, image_cv2_modified, 1-alpha, 0, image_cv2_modified)
                    seg_path_and_name_result = seg_im_out_path + 'seg_image_{:04d}'.format(im_idx) + '.jpg'
                    cv2.imwrite(seg_path_and_name_result, np_mask)
                    seg_path_and_name_result = seg_np_out_path + 'seg_image_{:04d}'.format(im_idx) + '.npy'
                    np.save(seg_path_and_name_result, np_mask, allow_pickle=False)
            
            cv2.imwrite(img_path_and_name_result, image_cv2_modified)
            im_idx += 1
            continue

        bag_in.close()
        print('Done with bag!')
        if im_idx > 0:
            ave_time /= im_idx # already has extra +1 to account for 0 indexing
            print("Average detection time = {:.3f} ms, maximum detection time = {:.3f} ms  ({:d} images)".format(ave_time * 1000, max_time * 1000, im_idx))
        else:
            print("WARNING: No images in rosbag with topic {}".format(topic_str))


if __name__ == '__main__':
    np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
    try:
        segment_object_pixels()
    except:
        import traceback
        traceback.print_exc()
    print("--------------- FINISHED ---------------")
