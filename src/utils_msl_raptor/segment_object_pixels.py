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


class segment_object_pixels:

    def __init__(self):
        # INPUTS
        max_num_images = -1  # set to -1 to have no limit
        image_skip_rate = 1  # set to 1 to use every image, 2 to use every other, etc etc
        b_save_images_with_masks = False
        b_output_debug_image = False
        object_name = "bowl_green_msl"   #    # "bottle_swell_1"  # "bowl_green_msl"  # "bowl_grey_msl"
        topic_str = ['/quad7/camera/image_raw', '/vrpn_client_node/quad7/pose', "/vrpn_client_node/" + object_name  + "/pose"]
        rb_path = '/mounted_folder/bags_to_test_coral_detect/'
        if object_name == "bowl_grey_msl":
            # GREY BOWL: 285, 275  | 395, 275  | 395, 330  | 285, 330
            rb_name = 'grey_bowl_msl/bowl_grey_msl_nerf_with_markers.bag'
            x_range = (285, 395) # min and max x pixel for image-aligned bounding box
            y_range = (275, 330) # min and max y pixel for image-aligned bounding box
        elif object_name == "bowl_green_msl":
            # GREEN BOWL: 290, 255  | 335, 255  | 335, 277  | 290, 277
            # rb_name = 'green_bowl_msl/bowl_green_msl_nerf_with_markers.bag'  
            # x_range = (290, 335) # min and max x pixel for image-aligned bounding box
            # y_range = (255, 277) # min and max y pixel for image-aligned bounding box
            # GREEN BOWL CLOSE: 244, 302  | 426, 302  | 426, 407  | 244, 407
            rb_name = 'green_bowl_msl_close/bowl_green_msl_nerf_with_markers.bag'  
            x_range = (244, 426) # min and max x pixel for image-aligned bounding box
            y_range = (302, 407) # min and max y pixel for image-aligned bounding box
        elif object_name == "bottle_swell_1":
            # BOTTLE SWELL 1: 244, 302  | 426, 302  | 426, 407  | 244, 407
            rb_name = 'bottle_swell_1/bottle_swell_1_nerf_with_markers.bag'
            x_range = (268, 338) # min and max x pixel for image-aligned bounding box
            y_range = (129, 394) # min and max y pixel for image-aligned bounding box
        else:
            raise RuntimeError("Object name not recognized")
        ########################################
        init_bb = ((x_range[0] + x_range[1])/2, (y_range[0] + y_range[1])/2, x_range[1] - x_range[0], y_range[1] - y_range[0])  # box format: x,y,w,h (where x,y correspond to the center of the bounding box)
        init_bb_up_left = (x_range[0], y_range[0], x_range[1] - x_range[0], y_range[1] - y_range[0])  # box format: x,y,w,h (where x,y correspond to the top left corner)
        
        base_directory_path = rb_path + rb_name[:-4] + '_output/'
        my_dirs = self.construct_directory_structure(base_directory_path, b_save_images_with_masks)
        T_cam_ego, T_ego_cam = self.construct_camera_transform() # note this is using hardcoded values calculated offline

        
        # definitions for loop
        self.bridge = CvBridge()
        b_first_loop = True
        im_idx = 0
        im_save_idx = 0
        im_times = []
        quad_pose_times = []
        obj_pose_times = []
        quad_poses = []
        obj_poses = []
        ave_time = 0
        max_time = 0
        bag_in = rosbag.Bag(rb_path + rb_name, 'r')

        # # first read the camera instrinsics and distortion params so we can undistort the images as they are read
        # for topic, msg, t in bag_in.read_messages(topics='/quad7/camera/camera_info'):
        #     K = np.reshape(msg.K, (3, 3))
        #     dist_coefs = np.reshape(msg.D, (5,))
        #     K_undistorted, _ = cv2.getOptimalNewCameraMatrix(K, dist_coefs, (msg.width, msg.height), 0, (msg.width, msg.height))
        #     K_3_4 = np.concatenate((K_undistorted.T, np.zeros((1,3)))).T
        #     break
        K = np.array([[486.12588397  , 0.   ,      328.90870824],
                        [  0.  ,       486.18350238, 250.9030576 ],
                        [  0. ,          0. ,          1.        ]])
        dist_coefs = np.array([-0.4489306 ,  0.28033736 ,-0.00006914,  0.00007105 , 0.11475715])
        K_undistorted, _ = cv2.getOptimalNewCameraMatrix(K, dist_coefs, (640,480), 0, (640,480))
        K_3_4 = np.concatenate((K_undistorted.T, np.zeros((1,3)))).T

        for topic, msg, t in bag_in.read_messages(topics=topic_str):
            if topic == '/vrpn_client_node/quad7/pose':
                quad_pose_times.append(t)
                quad_poses.append(pose_to_tf(msg.pose))
                continue
            elif topic == "/vrpn_client_node/" + object_name  + "/pose":
                obj_pose_times.append(t)
                obj_poses.append(pose_to_tf(msg.pose))
                continue
            elif topic == '/quad7/camera/image_raw':
                if im_idx % image_skip_rate > 0:
                    im_idx += 1
                    continue
                im_times.append(t)
                image_cv2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                image_cv2 = cv2.undistort(image_cv2, K, dist_coefs, None, K_undistorted)
                img_path_and_name = my_dirs["images"] + 'image_{:04d}'.format(im_save_idx) + '.jpg'
                cv2.imwrite(img_path_and_name, image_cv2)
                
                if b_first_loop:
                    b_first_loop = False
                    self.tracker = SiammaskTracker(image_cv2, base_dir='/root/msl_raptor_ws/src/msl_raptor/src/front_end/', use_tensorrt=False)
                    box0 = np.int0(cv2.boxPoints( ( (init_bb[0], init_bb[1]), (init_bb[2], init_bb[3]), 0)) )
                    next_state = self.tracker.reinit(init_bb_up_left, image_cv2)
                    redImg = np.zeros(image_cv2.shape, image_cv2.dtype)
                    redImg[:,:] = (0, 0, 255)
                    
                next_state, abb, mask = self.tracker.track(image_cv2, next_state)
                seg_path_and_name_result = my_dirs["seg_masks"] + 'seg_image_{:04d}'.format(im_save_idx) + '.npy'
                np.save(seg_path_and_name_result, mask, allow_pickle=False)

                if b_save_images_with_masks:
                    abb = bb_corners_to_angled_bb(abb.reshape(-1,2))
                    box = np.int0(cv2.boxPoints( ( (abb[0], abb[1]), (abb[2], abb[3]), -np.degrees(abb[4]))) )
                    image_cv2_modified = copy(image_cv2)
                    if b_first_loop:
                        cv2.drawContours(image_cv2_modified, [box0], 0, (255,0,0), 2)  # draw init box
                    cv2.drawContours(image_cv2_modified, [box], 0, (0,255,0), 2)
                    
                    np_mask = np.array(mask, dtype=np.uint8)
                    redMask = cv2.bitwise_and(redImg, redImg, mask=np_mask)
                    alpha = 0.3
                    cv2.addWeighted(redMask, alpha, image_cv2_modified, 1-alpha, 0, image_cv2_modified)
                    # seg_path_and_name_result = seg_im_out_path + 'seg_image_{:04d}'.format(im_save_idx) + '.jpg'
                    # cv2.imwrite(seg_path_and_name_result, np_mask)
                    img_path_and_name_result = my_dirs["masked_images"] + 'masked_image_{:04d}'.format(im_save_idx) + '.jpg'
                    cv2.imwrite(img_path_and_name_result, image_cv2_modified)

                im_idx += 1
                im_save_idx += 1
                if max_num_images > 0 and im_save_idx >= max_num_images:
                    break
        bag_in.close()

        im_save_idx = 0
        for im_time in im_times:
            quad_pose, _ = find_closest_by_time(time_to_match=im_time, time_list=quad_pose_times, message_list=quad_poses)
            T_w_obj, _ = find_closest_by_time(time_to_match=im_time, time_list=obj_pose_times, message_list=obj_poses)
            T_w_obj[2,3] = (67.5/1000)/2  # shift the origin of the bowl to the center of it's bounding box. It's dims are 1770 x 170, x 67.5 (mm)

            cam_pose_name = my_dirs["camera_poses"] + 'cam_pose_{:04d}'.format(im_save_idx) + '.npy'
            obj_pose_name = my_dirs["bowl_green_msl_poses"] + 'obj_pose_{:04d}'.format(im_save_idx) + '.npy'
            T_w_ego = quad_pose
            T_w_cam = T_w_ego @ T_ego_cam
            np.save(cam_pose_name, T_w_cam, allow_pickle=False)
            np.save(obj_pose_name, T_w_obj, allow_pickle=False)

            K_3_4_openGL = copy(K_3_4)
            K_3_4_openGL[0,2] = 640 - K_3_4_openGL[0,2]
            K_3_4_openGL[1,2] = 480 - K_3_4_openGL[1,2]
            K_3_4_openGL[0:3, 2] *= -1
            cam_proj_matK_3_4_openGL = K_3_4_openGL @ T_cam_ego @ inv_tf(T_w_ego) 
            cam_proj_path_and_name = my_dirs["projection_matrices"] + 'projection_matrix_{:04d}'.format(im_save_idx) + '.npy'
            np.save(cam_proj_path_and_name, cam_proj_matK_3_4_openGL, allow_pickle=False)

            if b_output_debug_image and im_save_idx==0:
                cam_proj_mat = K_3_4 @ T_cam_ego @ inv_tf(T_w_ego) 
                T_cam_w = inv_tf(T_w_cam)

                T_cam_obj = T_cam_w @ T_w_obj
                pnt_c = np.concatenate((T_cam_obj[0:3, 3], [1]))
                
                org_px = K_3_4 @ pnt_c
                (x, y) = np.round(np.array([org_px[0], org_px[1]]) / org_px[2]).astype(np.int)  # pixel of origin of bowl in image:  313 from left, 270 from top
                image_cv2 = cv2.circle(image_cv2, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

                pnt_w = np.concatenate((T_w_obj[0:3, 3], [1]))
                world_origin_px = cam_proj_mat @ pnt_w
                world_origin_px = np.array([world_origin_px[0], world_origin_px[1]]) / world_origin_px[2]
                (x, y) = np.round(world_origin_px).astype(np.int)  # note I am not sure if world_origin_px is (row, col) or (col,row)
                x = 640 - x  # WHY???
                image_cv2 = cv2.circle(image_cv2, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

                cv2.imwrite("/mounted_folder/testtestest.png", image_cv2)

            cam_fov_x = 2 * np.arctan(640/(2*K[0,0]))
            cam_fov_y = 2 * np.arctan(480/(2*K[1,1]))
            
            im_save_idx += 1

        pdb.set_trace()

        # find corresponding time to find which poses correspond to which images
        print('Done with bag!')
        if im_idx > 0:
            ave_time /= im_idx # already has extra +1 to account for 0 indexing
            print("Average detection time = {:.3f} ms, maximum detection time = {:.3f} ms  ({:d} images)".format(ave_time * 1000, max_time * 1000, im_idx))
        else:
            print("WARNING: No images in rosbag with topic {}".format(topic_str))


    def construct_directory_structure(self, base_directory_path, b_save_images_with_masks=False):
        my_dir_dict = {}
        my_dir_dict["base"] = base_directory_path
        my_dir_dict["images"] = base_directory_path + 'images/'
        my_dir_dict["seg_masks"] = base_directory_path + 'seg_masks/'
        my_dir_dict["camera_poses"] = base_directory_path + 'camera_poses/'
        my_dir_dict["bowl_green_msl_poses"] = base_directory_path + 'bowl_green_msl_poses/'
        my_dir_dict["projection_matrices"] = base_directory_path + 'projection_matrices/'
        if b_save_images_with_masks:
            my_dir_dict["masked_images"] = base_directory_path + 'masked_images/'

        for path in my_dir_dict:
            if not os.path.exists(my_dir_dict[path]):
                os.mkdir(my_dir_dict[path])
        print("done making directory structure")
        return my_dir_dict

    def construct_camera_transform(self):
        # this assumes camera axis is +z and y is down. OpenGL assumes cam axis is -z and y is up
        # R_cam_ego = np.reshape([-0.0246107,  -0.99869617, -0.04472445,  -0.05265648,  0.0459709,  -0.99755399, 0.99830938, -0.02219547, -0.0537192], (3,3))
        # t_cam_ego = np.asarray([0.11041654, 0.06015242, -0.07401183])
        R_cam_ego = np.reshape([ 0.02495941, -0.99952024, -0.01833889, -0.04791799,  0.01712734, -0.99870442,  0.99853938,  0.02580584, -0.04746751], (3,3))
        t_cam_ego = np.array([ 0.012847,    0.0223505,  -0.08445964])
        
        T_cam_ego = np.eye(4)
        T_cam_ego[0:3, 0:3] = R_cam_ego
        T_cam_ego[0:3, 3] = t_cam_ego
        T_ego_cam = np.eye(4)
        T_ego_cam[0:3, 0:3] = R_cam_ego.T
        T_ego_cam[0:3, 3] = -R_cam_ego.T @ t_cam_ego

        # T_cam_camGL = np.eye(4)
        T_cam_camGL = np.array([[ 1,  0,  0, 0],
                                [ 0, -1,  0, 0],
                                [ 0,  0, -1, 0],
                                [ 0,  0,  0, 1]])
        T_ego_cam = T_ego_cam @ T_cam_camGL
        T_cam_ego = inv_tf(T_cam_camGL) @ T_cam_ego
        return T_cam_ego, T_ego_cam

if __name__ == '__main__':
    np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
    try:
        segment_object_pixels()
    except:
        import traceback
        traceback.print_exc()
    print("--------------- FINISHED ---------------")
