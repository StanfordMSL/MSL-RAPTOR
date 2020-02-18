#!/usr/bin/env python3

# IMPORTS
# system
import os, sys, argparse, time
import pdb
# math
import numpy as np
import cv2
# ros
import rospy
# custom modules
from ros_interface import ros_interface as ROS
from ukf import UKF
# libs & utils
from utils_msl_raptor.ros_utils import *
from utils_msl_raptor.math_utils import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src/front_end')
from image_segmentor import ImageSegmentor
import yaml

def run_execution_loop():
    b_use_gt_bb = rospy.get_param('~b_use_gt_bb') 
    b_verbose = rospy.get_param('~b_verbose') 
    detection_period_ros = rospy.get_param('~detection_period') # In seconds
    objects_sizes_yaml = rospy.get_param('~object_sizes_file')
    objects_used_path = rospy.get_param('~object_used_file')
    classes_names_file = rospy.get_param('~classes_names_file')
    b_filter_meas = True
    b_axis_aligned_bb = True
    
    ros = ROS(b_use_gt_bb,b_verbose)  # create a ros interface object

    bb_3d, obj_width, classes_names, classes_ids = init_objects(objects_sizes_yaml,objects_used_path,classes_names_file)  # Parse objects used and associated configurations

    if b_use_gt_bb:
        rospy.logwarn("\n\n\n------------- IN DEBUG MODE (Using Ground Truth Bounding Boxes) -------------\n\n\n")
        time.sleep(0.5)
    
    if not b_use_gt_bb:
        print('Waiting for first image')
        im = ros.get_first_image()
        print('initializing image segmentor!!!!!!')
        ros.im_seg = ImageSegmentor(im,use_trt=rospy.get_param('~b_use_tensorrt'), detection_period=detection_period_ros,verbose=b_verbose,detect_classes_ids=classes_ids,detect_classes_names=classes_names, b_axis_aligned_bb=b_axis_aligned_bb)
        print('initializing DONE - PLAY BAG NOW!!!!!!')
        time.sleep(0.5)
    
    rate = rospy.Rate(30) # max filter rate
    ukf_dict = {}  # key: object_id value: ukf object
    ros.camera = camera(ros)
    my_camera = ros.camera

    ros.create_subs_and_pubs()
    dim_state = 13
    state_est = np.zeros((dim_state + dim_state**2, ))
    loop_count = 0
    previous_image_time = 0

    if b_use_gt_bb:
        init_state_from_gt(ros, ukf_dict['quad4'])  
        img_seg = None

    tic = time.time()
    while not rospy.is_shutdown():
        # store data locally (so it doesnt get overwritten in ROS object)
        loop_time = ros.latest_img_time
        if loop_time <= previous_image_time or ros.latest_img_time < 0:
            # this means we dont have new data yet
            rate.sleep()
            continue
        if loop_count == 0:
            # first iteration, need initial time so dt will be accurate
            previous_image_time = loop_time
            loop_count += 1
            rate.sleep()
            continue
        
        # get latest data from ros
        processed_image = ros.im_process_output
        im_seg_mode = ros.latest_bb_method

        # do we have any objects?
        num_obj_in_img = len(processed_image)
        if num_obj_in_img == 0:  # if no objects are seen, dont do anything
            print("No objects detected/tracked in FOV")
            rate.sleep()
            continue
        
        tf_w_ego = ros.tf_w_ego
        tf_ego_w = inv_tf(tf_w_ego)  # ego quad pose
        
        if b_use_gt_bb:
            raise RuntimeError("b_use_gt_bb option NOT YET IMPLEMENTED")

        # handle each object seen
        obj_ids_tracked = []
        for obj_id, (abb, class_str, valid) in processed_image.items():
            ukf = None
            if not obj_id in ukf_dict:  # New Object
                print("new object (id = {}, type = {})".format(obj_id, class_str))
                ukf_dict[obj_id] = UKF(camera=my_camera, bb_3d=bb_3d[class_str], obj_width=obj_width[class_str], init_time=loop_time, class_str=class_str, obj_id=obj_id,verbose=b_verbose, b_axis_aligned_bb=b_axis_aligned_bb)
                ukf_dict[obj_id].reinit_filter(abb, tf_w_ego)
                continue

            obj_ids_tracked.append(obj_id)

            previous_image_time = loop_time  # this ensures we dont reuse the image

            if ukf_dict[obj_id] is not None:
                ukf_dict[obj_id].step_ukf(abb, tf_ego_w, loop_time)  # update ukf
        
        ros.publish_filter_state(obj_ids_tracked,ukf_dict)#, ukf_dict[obj_id].mu, ukf_dict[obj_id].itr_time, ukf_dict[obj_id].itr)  # send vector with time, iteration, state_est
        ros.publish_bb_msg(processed_image,im_seg_mode, loop_time)# obj_ids_tracked, abb, im_seg_mode, loop_time)
        
        # Save current object states in image segmentor
        ros.im_seg.ukf_dict = ukf_dict

        # ros.im_seg_mode = ros.TRACK
        if b_verbose:
            print("FULL END-TO-END time = {:4f}\n".format(time.time() - tic))
        rate.sleep()
        tic = time.time()
        loop_count += 1
    ### DONE WITH MSL RAPTOR ####


def init_state_from_gt(ros, ukf):
    # init ukf state
    rospy.logwarn('using ground truth to initialize filter!')
    ukf.mu = pose_to_state_vec(ros.ado_pose_gt_rosmsg) 
    ukf.mu[0:3] += np.array([-2, .5, .5]) 


def init_objects(objects_sizes_yaml,objects_used_path,classes_names_file):
    # create camera object (see https://github.com/StanfordMSL/uav_game/blob/tro_experiments/ec_quad_sim/ec_quad_sim/param/quad3_trans.yaml)

    with open(objects_used_path) as f:
        objects_used = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    objects_used = [x.strip() for x in objects_used]

    with open(classes_names_file) as f:
        classes_names = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    classes_names = [x.strip() for x in classes_names]
    classes_used_names = []
    classes_used_ids = []
    bb_3d = {}
    obj_width = {}
    with open(objects_sizes_yaml, 'r') as stream:
        try:
            obj_prms = list(yaml.load_all(stream))
            for obj_dict in obj_prms:
                if obj_dict['ns'] in objects_used:
                    half_length = float(obj_dict['bound_box_l']) /2
                    half_width = float(obj_dict['bound_box_w']) /2
                    half_height = float(obj_dict['bound_box_h']) /2
                    
                    bb_3d[obj_dict['class_str']] = np.array([[ half_length, half_width, half_height, 1.],  # 1 front, left,  up (from quad's perspective)
                          [ half_length, half_width,-half_height, 1.],  # 2 front, right, up
                          [ half_length,-half_width,-half_height, 1.],  # 3 back,  right, up
                          [ half_length,-half_width, half_height, 1.],  # 4 back,  left,  up
                          [-half_length,-half_width, half_height, 1.],  # 5 front, left,  down
                          [-half_length,-half_width,-half_height, 1.],  # 6 front, right, down
                          [-half_length, half_width,-half_height, 1.],  # 7 back,  right, down
                          [-half_length, half_width, half_height, 1.]]) # 8 back,  left,  down

                    obj_width[obj_dict['class_str']] = 2*half_width

                    # Add the object's class to the classes used, if not there already
                    if obj_dict['class_str'] not in classes_names:
                        print('Class '+obj_dict['class_str']+' not found in the list of classes supported')
                    elif obj_dict['class_str'] not in classes_used_names:
                        classes_used_names.append(obj_dict['class_str'])
                        classes_used_ids.append(classes_names.index(obj_dict['class_str']))

        except yaml.YAMLError as exc:
            print(exc)

    return bb_3d, obj_width, classes_used_names, classes_used_ids


def wait_intil_ros_ready(ros, rate):
    """ pause until ros is ready or timeout reached """
    rospy.loginfo("waiting for ros...")
    while ros.latest_img_time is None:
        rate.sleep()
        continue
    rospy.loginfo("ROS is initialized!")


class camera:
    def __init__(self, ros):
        """
        K: camera intrinsic matrix 
        tf_cam_ego: camera pose relative to the ego_quad (fixed)
        fov_horz/fov_vert: Angular field of view (IN RADIANS) for horizontal and vertical directions
        fov_lim_per_depth: how the boundary of the fov (width, heigh) changes per depth
        """
        ns = rospy.get_param('~ns')
        camera_info = rospy.wait_for_message(ns + '/camera/camera_info', CameraInfo,500)
        self.K = np.reshape(camera_info.K, (3, 3))
        if len(camera_info.D) == 5:
            self.dist_coefs = np.reshape(camera_info.D, (5,))
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (camera_info.width, camera_info.height), 0, (camera_info.width, camera_info.height))
        else:
            self.dist_coefs = None
            self.new_camera_matrix = self.K

        self.K_inv = la.inv(self.K)
        self.new_camera_matrix_inv = la.inv(self.new_camera_matrix)
        self.tf_cam_ego = np.eye(4)
        self.tf_cam_ego[0:3, 3] = np.asarray(rospy.get_param('~t_cam_ego'))
        self.tf_cam_ego[0:3, 0:3] = np.reshape(rospy.get_param('~R_cam_ego'), (3, 3))
        (self.fov_horz, self.fov_vert), self.fov_lim_per_depth = self.calc_fov()

    def b_is_pnt_in_fov(self, pnt_c, buffer=0):
        """ 
        - Use similar triangles to see if point (in camera frame!) is beyond limit of fov 
        - buffer: an optional buffer region where if you are inside the fov by less than 
            this the function returns false
        """
        if pnt_c[2] <= 0:
            raise RuntimeError("Point is at or behind camera!")
            return False
        fov_lims = pnt_c[2] * self.fov_lim_per_depth - buffer
        return np.all( np.abs(pnt_c[0:2]) < fov_lims )

    def calc_fov(self):
        """
        - Find top, left point 1 meter along z axis in cam frame. the x and y values are 
        half the width and height. Note: [x_tl, y_tl, 1 (= z_tl)] = inv(K) @ [0, 0, 1], 
        which is just the first tow rows of the third col of inv(K).
        - With these x and y, just use geometry (knowing z dist is 1) to get the angle 
        spanning the x and y axis respectively.
        - keeping the width and heigh of the point at 1m depth is useful for similar triangles
        """
        fov_lim_per_depth = -la.inv( self.new_camera_matrix )[0:2, 2] 
        return 2 * np.arctan( fov_lim_per_depth ), fov_lim_per_depth

    def pix_to_pnt3d(self, row, col):
        """
        input: assumes rc is [row, col]
        output: pnt_c = [x, y, z] in camera frame
        """
        raise RuntimeError("FUNCTION NOT YET IMPLEMENTED")
        return pnt_c

    def pnt3d_to_pix(self, pnt_c):
        """
        input: assumes pnt in camera frame
        output: [row, col] i.e. the projection of xyz onto camera plane
        """
        rc = self.new_camera_matrix @ np.reshape(pnt_c[0:3], 3, 1)
        rc = np.array([rc[1], rc[0]]) / rc[2]
        return rc


if __name__ == '__main__':
    np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
    print("Starting MSL-RAPTOR main [running python {}]".format(sys.version_info[0]))
    rospy.init_node('RAPTOR_MSL', anonymous=True)
    run_execution_loop()
    print("--------------- FINISHED ---------------")

