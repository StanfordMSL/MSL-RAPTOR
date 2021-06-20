#!/usr/bin/env python3
import sys, os, time
import pdb

import numpy as np

import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
sys.path.insert(1, '/root/msl_raptor_ws/src/msl_raptor/src/front_end/coral/tflite/python/examples/detection')
# sys.path.append(os.path.dirname(os.path.dirname('/root/msl_raptor_ws/src/msl_raptor/src/front_end/coral/tflite/python/examples/detection')))
import detect_image_coral
import detect_coral


import argparse
import time
from PIL import Image
from PIL import ImageDraw
import tflite_runtime.interpreter as tflite
import platform


class rosbag_object_detector:
    def __init__(self):
        self.bridge = CvBridge()
        rb_path = '/mounted_folder/bags_to_test_coral_detect/'
        # rb_name = '2021-06-12-20-12-00.bag'
        # rb_name = '2021-06-12-20-29-39.bag'
        rb_name = '2021-06-20-12-42-38.bag'
        img_out_path = rb_path + rb_name[:-4] + '_output/'
        if not os.path.exists(img_out_path):
             os.mkdir(img_out_path)
        img_det_out_path = img_out_path + 'img_with_detection/'
        if not os.path.exists(img_det_out_path):
             os.mkdir(img_det_out_path)

        bag_in = rosbag.Bag(rb_path + rb_name, 'r')

        label_file = '/mounted_folder/models/coco_labels.txt'
        model_file = '/mounted_folder/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'

        labels = detect_image_coral.load_labels(label_file) if label_file else {}
        interpreter = detect_image_coral.make_interpreter(model_file)
        interpreter.allocate_tensors()
        scale = None
        thresh = 0.4
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
            cv2.imwrite(img_path_and_name, image_cv2)
            
            img_path_and_name_result = img_det_out_path + 'image_result_{:04d}'.format(im_idx) + '.jpg'
            image = Image.open(img_path_and_name)
            
            if b_first_loop:
                b_first_loop = False
                start = time.perf_counter()
                scale = detect_coral.set_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
                interpreter.invoke()
                objs = detect_coral.get_output(interpreter, thresh, scale)  # call this once in begining which will take longer as model is loaded onto device
                print("Time to load model onto device: {:.2f} ms".format((time.perf_counter() - start)*1000))

            start = time.perf_counter()
            scale = detect_coral.set_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
            interpreter.invoke()
            objs = detect_coral.get_output(interpreter, thresh, scale)
            inference_time = time.perf_counter() - start
            ave_time += inference_time
            if inference_time > max_time:
                max_time = inference_time
            # print('%.2f ms' % (inference_time * 1000))
            
            if b_save_output:
                image = image.convert('RGB')
                if objs:
                    detect_image_coral.draw_objects(ImageDraw.Draw(image), objs, labels)
                image.save(img_path_and_name_result)
                # image.show()
            im_idx += 1
            # pdb.set_trace()
            objs = None  # reset
            
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
        rosbag_object_detector()
    except:
        import traceback
        traceback.print_exc()
    print("--------------- FINISHED ---------------")
