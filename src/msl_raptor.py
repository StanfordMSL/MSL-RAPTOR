#!/usr/bin/env python

"""
The MIT License (MIT)
Copyright (c) 2019 MSL
"""

# IMPORTS
# system
import os, sys, argparse, time
# from pathlib import Path
# save/load
# import pickle
# math
import numpy as np
# plots
# import matplotlib
# from matplotlib import pyplot as plt
# from mpl_toolkits import mplot3d
# libs
from ros_interface import ros_interface as ROS

def run_execution_loop():
    b_target_in_view = True
    ros = ROS()

    loop_count = 0
    while b_target_in_view:
        # read in latest data
        # update ukf
        # [optional] update plots
        loop_count += 1
        if loop_count > 100:
            break

if __name__ == '__main__':
    print("Starting MSL-RAPTOR main")
    run_execution_loop()
    print("--------------- FINISHED ---------------")