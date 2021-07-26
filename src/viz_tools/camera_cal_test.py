#!/usr/bin/env python3
# IMPORTS
# system
import sys, time
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

class camera_cal_test_node:
    """
    This rosnode has two modes. In mode 1 it publishes a white background with the bounding boxes drawn (green when tracking, red when detecting). 
    This is faster and less likely to slow down the network. Mode 2 publishes the actual image. This is good for debugging, but is slower.
    If rosparam b_overlay is false (default), it will be mode 1, else mode 2.
    """

    def __init__(self):
        rospy.init_node('camera_cal_test_node', anonymous=True)
        self.bridge = CvBridge()
        self.ns = rospy.get_param('~ns')  # robot namespace
        
        camera_info = rospy.wait_for_message(self.ns + '/camera/camera_info', CameraInfo, 30)
        if False:
            self.K = np.reshape(camera_info.K, (3, 3))
            if len(camera_info.D) == 5:
                self.dist_coefs = np.reshape(camera_info.D, (5,))
                self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (camera_info.width, camera_info.height), 0, (camera_info.width, camera_info.height))
            else:
                self.dist_coefs = None
                self.new_camera_matrix = self.K
        else:
           self.dist_coefs = np.array([-0.40031982,  0.14257124,  0.00020686,  0.00030526,  0.        ])
           self.K = np.array([[483.50426183,   0.        , 318.29104565],
                              [  0.        , 483.89448247, 248.02496288],
                              [  0.        ,   0.        ,   1.        ]])
           self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coefs, (camera_info.width, camera_info.height), 0, (camera_info.width, camera_info.height))

        self.K_3_4_undistorted = np.concatenate((self.new_camera_matrix.T, np.zeros((1,3)))).T
        rospy.Subscriber(self.ns + '/camera/image_raw', Image, self.image_cb, queue_size=1, buff_size=2**21)
        self.img_overlay_pub = rospy.Publisher(self.ns + '/image_bb_overlay', Image, queue_size=5)

        self.img_buffer = ([], [])
        self.img_rosmesg_buffer_len = 500

        self.ego_pose_rosmesg_buffer = ([], [])
        self.ego_pose_rosmesg_buffer_gt = ([], [])
        self.ego_pose_rosmesg_buffer_len = 50
        self.ego_pose_gt_rosmsg = None

        self.ado_pose_rosmesg_buffer = ([], [])
        self.ado_pose_rosmesg_buffer_gt = ([], [])
        self.ado_pose_rosmesg_buffer_len = 50
        self.ado_pose_gt_rosmsg = None

        test_ego_object_name = 'quad7'
        # test_ego_object_name = 'AAA'
        test_ego_object_topic = '/vrpn_client_node/' + test_ego_object_name + '/pose'
        rospy.Subscriber(test_ego_object_topic, PoseStamped, self.ego_pose_gt_cb, queue_size=10)  # optitrack pose

        test_ado_object_name = 'bowl_green_msl'
        # test_ado_object_name = 'BBB'
        test_ado_object_topic = '/vrpn_client_node/' + test_ado_object_name + '/pose'
        rospy.Subscriber(test_ado_object_topic, PoseStamped, self.ado_pose_gt_cb, queue_size=10)
        self.ado_3d_bb_dims = np.array([170, 170, 67.5])/1000  # x, y, z dim in meters (local frame)


        self.R_cam_ego = np.reshape([-0.0246107,  -0.99869617, -0.04472445,  -0.05265648,  0.0459709,  -0.99755399, 0.99830938, -0.02219547, -0.0537192], (3,3))
        self.t_cam_ego = np.asarray([0.11041654, 0.06015242, -0.07401183])
        
        self.T_cam_ego = np.eye(4)
        self.T_cam_ego[0:3, 0:3] = self.R_cam_ego
        self.T_cam_ego[0:3, 3] = self.t_cam_ego
        self.T_ego_cam = np.eye(4)
        self.T_ego_cam[0:3, 0:3] = self.R_cam_ego.T
        self.T_ego_cam[0:3, 3] = -self.R_cam_ego.T @ self.t_cam_ego

    
    def ado_pose_gt_cb(self, msg):
        """
        Maintains a buffer of poses and times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """
        my_time = get_ros_time(msg)  # time in seconds

        if len(self.ado_pose_rosmesg_buffer_gt[0]) < self.ado_pose_rosmesg_buffer_len:
            self.ado_pose_rosmesg_buffer_gt[0].append(msg.pose)
            self.ado_pose_rosmesg_buffer_gt[1].append(my_time)
        else:
            self.ado_pose_rosmesg_buffer_gt[0][0:self.ado_pose_rosmesg_buffer_len] = self.ado_pose_rosmesg_buffer_gt[0][1:self.ado_pose_rosmesg_buffer_len]
            self.ado_pose_rosmesg_buffer_gt[1][0:self.ado_pose_rosmesg_buffer_len] = self.ado_pose_rosmesg_buffer_gt[1][1:self.ado_pose_rosmesg_buffer_len]
            self.ado_pose_rosmesg_buffer_gt[0][-1] = msg.pose
            self.ado_pose_rosmesg_buffer_gt[1][-1] = my_time

    def ego_pose_gt_cb(self, msg):
        # self.ego_pose_gt_rosmsg = msg.pose
        """
        Maintains a buffer of poses and times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """
        my_time = get_ros_time(msg)  # time in seconds

        if len(self.ego_pose_rosmesg_buffer_gt[0]) < self.ego_pose_rosmesg_buffer_len:
            self.ego_pose_rosmesg_buffer_gt[0].append(msg.pose)
            self.ego_pose_rosmesg_buffer_gt[1].append(my_time)
        else:
            self.ego_pose_rosmesg_buffer_gt[0][0:self.ego_pose_rosmesg_buffer_len] = self.ego_pose_rosmesg_buffer_gt[0][1:self.ego_pose_rosmesg_buffer_len]
            self.ego_pose_rosmesg_buffer_gt[1][0:self.ego_pose_rosmesg_buffer_len] = self.ego_pose_rosmesg_buffer_gt[1][1:self.ego_pose_rosmesg_buffer_len]
            self.ego_pose_rosmesg_buffer_gt[0][-1] = msg.pose
            self.ego_pose_rosmesg_buffer_gt[1][-1] = my_time


    def image_cb(self, msg):
        """
        Maintains a buffer of images & times. The first element is the earliest. 
        Stored in a way to interface with a quick method for finding closest match by time.
        """
        # my_time = rospy.Time.now().to_sec()  # time in seconds
        my_time = msg.header.stamp.to_sec()  # time in seconds

        if len(self.img_buffer[0]) < self.img_rosmesg_buffer_len:
            self.img_buffer[0].append(msg)
            self.img_buffer[1].append(my_time)
        else:
            self.img_buffer[0][0:self.img_rosmesg_buffer_len] = self.img_buffer[0][1:self.img_rosmesg_buffer_len]
            self.img_buffer[1][0:self.img_rosmesg_buffer_len] = self.img_buffer[1][1:self.img_rosmesg_buffer_len]
            self.img_buffer[0][-1] = msg
            self.img_buffer[1][-1] = my_time


    def find_closest_by_time_ros2(self, time_to_match, time_list, message_list=None):
        """
        Assumes lists are sorted earlier to later. Returns closest item in list by time. If two numbers are equally close, return the smallest number.
        Adapted from https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
        """
        if not message_list:
            message_list = time_list
        pos = bisect_left(time_list, time_to_match)
        if pos == 0:
            t_delta = time_list[0]- time_to_match
            if t_delta > 0.0001:
                print("warning: \"bottoming out\" in buffer (match time = {:.4f}, closest queue time = {:.4f}, delta = {:.4f})".format(time_to_match, time_list[0], t_delta))
            return message_list[0], 0
        if pos == len(time_list):
            return message_list[-1], len(message_list) - 1
        before = time_list[pos - 1]
        after = time_list[pos]
        if after - time_to_match < time_to_match - before:
            return message_list[pos], pos
        else:
            return message_list[pos - 1], pos - 1

    def run(self):
        rate = rospy.Rate(15)

        while not rospy.is_shutdown():
            # print("{} and {} and {}".format(len(self.img_buffer[0]), len(self.ego_pose_rosmesg_buffer_gt[0]), len(self.ado_pose_rosmesg_buffer_gt[0])))
            if len(self.img_buffer[0]) > 0 and len(self.ego_pose_rosmesg_buffer_gt[0]) > 0 and len(self.ado_pose_rosmesg_buffer_gt[0]) > 0:
                # get & undistort image
                img_msg = self.img_buffer[0][-1]
                open_cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
                open_cv_image = cv2.undistort(open_cv_image, self.K, self.dist_coefs, None, self.new_camera_matrix)

                if True:
                    # get pose data
                    img_time = self.img_buffer[1][-1]
                    T_w_ego_gt = pose_to_tf(self.find_closest_by_time_ros2(img_time, self.ego_pose_rosmesg_buffer_gt[1], self.ego_pose_rosmesg_buffer_gt[0])[0])
                    T_w_ado_gt = pose_to_tf(self.find_closest_by_time_ros2(img_time, self.ado_pose_rosmesg_buffer_gt[1], self.ado_pose_rosmesg_buffer_gt[0])[0])
                    T_w_ado_gt[2,3] = self.ado_3d_bb_dims[2]/2  # shift the origin of the bowl to the center of it's bounding box. It's dims are 170 x 170, x 67.5 (mm)
                    T_ego_ado = inv_tf(T_w_ego_gt) @ T_w_ado_gt
                    T_cam_ado = self.T_cam_ego @ T_ego_ado

                    # draw bounding box
                    # generate corners in ado frame
                    # corners = []
                    # for x in [-0.5, 0.5]:
                    #     for y in [-0.5, 0.5]:
                    #         for z in [-0.5, 0.5]:
                    #             corners.append(np.array([x, y, z, 1]))
                    # corners = np.stack(corners, axis=-1)
                    box_length, box_width, box_height = self.ado_3d_bb_dims
                    vertices = np.array([[ box_length/2, box_width/2, box_height/2, 1.],
                                         [ box_length/2, box_width/2,-box_height/2, 1.],
                                         [ box_length/2,-box_width/2,-box_height/2, 1.],
                                         [ box_length/2,-box_width/2, box_height/2, 1.],
                                         [-box_length/2,-box_width/2, box_height/2, 1.],
                                         [-box_length/2,-box_width/2,-box_height/2, 1.],
                                         [-box_length/2, box_width/2,-box_height/2, 1.],
                                         [-box_length/2, box_width/2, box_height/2, 1.]]).T


                    corners_cam_frame = T_cam_ado @ vertices
                    corners2D_scaled = self.K_3_4_undistorted @ corners_cam_frame
                    corners2D = np.empty((corners2D_scaled.shape[1], 2))
                    for idx, corner_scaled in enumerate(corners2D_scaled.T):
                        corners2D[idx, :] = np.asarray((corner_scaled[0], corner_scaled[1])/corner_scaled[2])


                    # connected_inds = np.array([[0, 1], [1, 2], [2, 3], [3, 0],  # edges of front surface of 3D bb (starting at "upper left" and going counter-clockwise while facing the way the object is)
                    #                            [7, 4], [4, 5], [5, 6], [6, 7],  # edges of back surface of 3D bb (starting at "upper left" and going counter-clockwise while facing the way the object is)
                    #                            [0, 7], [1, 6], [2, 5], [3, 4]]) # horizontal edges of 3D bb (starting at "upper left" and going counter-clockwise while facing the way the object is)

                    inds_to_connect = [[0, 3], [3, 2], [2, 1], [1, 0], [7, 4], [4, 5], 
                                    [5, 6], [6, 7], [3, 4], [2, 5], [0, 7], [1, 6]]
                    cv_image_with_box = draw_2d_proj_of_3D_bounding_box(open_cv_image, corners2D, corners2D_gt=None, color_pr=(0,0,255), linewidth=1, inds_to_connect=inds_to_connect, b_verts_only=False)
                    cv2.imwrite("/mounted_folder/testtestest4.png", cv_image_with_box)
                    # pdb.set_trace()
                else:
                    cv_image_with_box = open_cv_image
                    open_cv_image2 = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
                    cv2.imwrite("/mounted_folder/testtestest3.png", open_cv_image2)
                    cv2.imwrite("/mounted_folder/testtestest2.png", open_cv_image)
                
                self.img_overlay_pub.publish(self.bridge.cv2_to_imgmsg(cv_image_with_box, "bgr8"))
                # pdb.set_trace()
            rate.sleep()


if __name__ == '__main__':
    np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
    try:
        program = camera_cal_test_node()
        program.run()
    except:
        import traceback
        traceback.print_exc()
    print("--------------- FINISHED ---------------")



# # new thing
# array([-0.44591373,  0.27564584,  0.        ,  0.        , -0.112326  ])


# ## seg
# array([-0.40031982,  0.14257124,  0.00020686,  0.00030526,  0.        ])

# (Pdb) K
# array([[483.50426183,   0.        , 318.29104565],
#        [  0.        , 483.89448247, 248.02496288],
#        [  0.        ,   0.        ,   1.        ]])
# (Pdb) K_undistorted
# array([[381.80621338,   0.        , 317.82922388],
#        [  0.        , 430.04415894, 250.37380723],
#        [  0.        ,   0.        ,   1.        ]])