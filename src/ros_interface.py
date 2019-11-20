# IMPORTS
# system
import sys, time #, argparse
# math
import numpy as np
# ros
import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


class ros_interface:

    def __init__(self):
        rospy.init_node('bounding_box_network_interface', anonymous=True)
        self.VERBOSE = True

        # Paramters #############################
        self.b_detect_new_bb = True  # set to false if last frame we had a bb (if false, use a tracking network like SiamMask)
        self.latest_bb = None
        self.latest_time = 0.0
        self.latest_ego_pose = None
        self.latest_bb_method = -1  # -1 for detect network, 1 for tracking network
        ####################################################################
        
        # Subscribers / Listeners & Publishers #############################
        # self.trans_listener = tf.TransformListener()
        # rospy.Subscriber('/IMAGE_TOPIC', IMAGE_MSG, self.NN_CALLBACK)
        self.bb_pub = rospy.Publisher('/bb', Float32MultiArray, queue_size=5)
        ####################################################################
        
        # create dummy message to send #############################
        data_len = 5
        self.dummy_bb_msg = Float32MultiArray()
        self.dummy_bb_msg.layout.dim.append(MultiArrayDimension())
        self.dummy_bb_msg.layout.dim.append(MultiArrayDimension())
        self.dummy_bb_msg.layout.dim[0].size = data_len
        self.dummy_bb_msg.layout.dim[1].size = 1
        self.dummy_bb_msg.layout.dim[0].stride = data_len*1
        self.dummy_bb_msg.layout.dim[1].stride = data_len
        self.dummy_bb_msg.layout.dim[0].label = "rows"
        self.dummy_bb_msg.layout.dim[1].label = "cols"
        self.dummy_bb_msg.layout.data_offset = 0
        self.dummy_bb_msg.data = [self.latest_time, 120, 230, 40, 20, 10*np.pi/180, self.latest_bb_method]
        ####################################################################
        
        self.run()


    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.bb_pub.publish(self.dummy_bb_msg)
            rate.sleep()
