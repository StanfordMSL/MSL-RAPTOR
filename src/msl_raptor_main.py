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


def run_execution_loop():
    b_enforce_0_yaw = rospy.get_param('~b_enforce_0_yaw') 
    b_use_gt_bb = rospy.get_param('~b_use_gt_bb') 
    detection_period = rospy.get_param('~detection_period') # In seconds
    b_filter_meas = True
    
    ros = ROS(b_use_gt_bb)  # create a ros interface object

    if b_use_gt_bb:
        rospy.logwarn("\n\n\n------------- IN DEBUG MODE (Using Ground Truth Bounding Boxes) -------------\n\n\n")
        time.sleep(0.5)
    
        
    if not b_use_gt_bb:
        print('Waiting for first image')
        im = ros.get_first_image()
        print('initializing image segmentor!!!!!!')
        ros.img_seg = ImageSegmentor(im,use_trt=rospy.get_param('~b_use_tensorrt'))
        ros.im_seg_mode = ros.DETECT
        print('initializing DONE - PLAY BAG NOW!!!!!!')
        time.sleep(0.5)
    
    rate = rospy.Rate(30) # max filter rate
    # wait_intil_ros_ready(ros, rate)  # pause to allow ros to get initial messages
    ukf_list = []  # index in list = obect id
    # ukf = UKF(b_enforce_0_yaw, b_use_gt_bb, obj_type='mslquad', obj_id=0, ukf_init_values)  # create ukf object
    my_camera, bb_3d, quad_width = init_objects(ros)  # init camera, pose, etc
    ros.create_subs_and_pubs()
    dim_state = 13
    state_est = np.zeros((dim_state + dim_state**2, ))
    loop_count = 0
    previous_image_time = 0
    time_since_last_detection = 0.

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
            ros.im_seg_mode = ros.DETECT
            rate.sleep()
            continue
        
        tf_ego_w = inv_tf(ros.tf_w_ego)  # ego quad pose
        # dt = loop_time - previous_image_time
        
        if b_use_gt_bb:
            raise RuntimeError("b_use_gt_bb option NOT YET IMPLEMENTED")

        # handle each object seen
        for idx, abb, obj_type, obj_id in enumerate(processed_image):
            ukf = None
            if ros.latest_bb_method == ros.DETECT:
                time_since_last_detection = 0.
                if obj_id > len(ukf_list):  # New Object
                    new_ukf = UKF(camera=my_camera, bb_3d=bb_3d, quad_width=quad_width, init_time=loop_time, obj_type='mslquad', obj_id=obj_id)
                    ukf_list.append(new_ukf)
                    continue
                else:  # Previously seen (but redetected) object, next time TRACK
                    ukf = ukf_list[idx]
                    ros.im_seg_mode = ros.TRACK

            elif ros.latest_bb_method == ros.TRACK:
                # check measurement
                ukf = ukf_list[idx]
        
            if time_since_last_detection > detection_period:
                ros.im_seg_mode = ros.DETECT
                time_since_last_detection = 0.
            else:
                time_since_last_detection +=  loop_time - previous_image_time
            
            previous_image_time = loop_time  # this ensures we dont reuse the image

            if ukf is not None:
                ukf.itr_time = loop_time
                ukf.step_ukf(abb, tf_ego_w, loop_time)  # update ukf
        
            ros.publish_filter_state(obj_id, ukf.mu, ukf.itr_time, ukf.itr)  # send vector with time, iteration, state_est
            ros.publish_bb_msg(obj_id, abb, im_seg_mode, loop_time)
        
        # ros.im_seg_mode = ros.TRACK
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


def init_objects(ros):
    # create camera object (see https://github.com/StanfordMSL/uav_game/blob/tro_experiments/ec_quad_sim/ec_quad_sim/param/quad3_trans.yaml)
    ros.camera = camera(ros)

    # init 3d bounding box in quad frame
    half_length = rospy.get_param('~target_bound_box_l') / 2
    half_width = rospy.get_param('~target_bound_box_w') / 2
    half_height = rospy.get_param('~target_bound_box_h') / 2
    bb_3d = np.array([[ half_length, half_width, half_height, 1.],  # 1 front, left,  up (from quad's perspective)
                          [ half_length, half_width,-half_height, 1.],  # 2 front, right, up
                          [ half_length,-half_width,-half_height, 1.],  # 3 back,  right, up
                          [ half_length,-half_width, half_height, 1.],  # 4 back,  left,  up
                          [-half_length,-half_width, half_height, 1.],  # 5 front, left,  down
                          [-half_length,-half_width,-half_height, 1.],  # 6 front, right, down
                          [-half_length, half_width,-half_height, 1.],  # 7 back,  right, down
                          [-half_length, half_width, half_height, 1.]]) # 8 back,  left,  down

    quad_width = 2*half_width

    ros.im_seg_mode = ros.DETECT  # start off by detecting
    return ros.camera, bb_3d, quad_width


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
        camera_info = rospy.wait_for_message(ns + '/camera/camera_info', CameraInfo, 10)
        self.K = np.reshape(camera_info.K, (3, 3))
        self.dist_coefs = np.reshape(camera_info.D, (5,))
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (camera_info.width, camera_info.height), 0, (camera_info.width, camera_info.height))

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

