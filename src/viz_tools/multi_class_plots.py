#!/usr/bin/env python3
# IMPORTS
# system
import sys, os, time, glob
from copy import copy
import pdb
# math
import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R
# plots
import matplotlib
import matplotlib.pyplot as plt
# Utils
sys.path.append('/root/msl_raptor_ws/src/msl_raptor/src/utils_msl_raptor')
from raptor_logger import *

class MultiObjectPlotGenerator:

    def __init__(self, base_directory, class_labels):

        # class_names = glob.glob(base_directory)
        # all_sub_folders = os.listdir(base_directory)
        self.class_labels = class_labels
        self.base_dir = base_directory

        logger = RaptorLogger(mode="read", base_path=self.base_dir)
        err_log_dict = defaultdict(list)
        for cl in self.class_labels:
            sub_dir = self.base_dir + cl
            err_logs_list = glob.glob(sub_dir + "/*_err.log")
            err_logs_list.extend(glob.glob(sub_dir + "/*_ssperr.log"))
            for log_path in err_logs_list:
                logs = logger.read_err_logs(log_path=log_path)
                err_log_dict[cl].append(logs)
        # make plot with varying thresholds
        self.fig = None
        self.axes = None
        color_strs = ['r', 'b', 'm', 'k', 'c', 'g']
        if len(err_log_dict) > len(color_strs):
            raise RuntimeError("need to add more colors to color string list!! too many classes..")


        nx = 100  # number of ticks on x axis
        dist_thresh = np.linspace(0, 0.2,nx)
        ang_thresh = np.linspace(0, 60,nx)
        thresh_list = list(zip(dist_thresh, ang_thresh)) # [(m thesh, deg thresh)]
        show_every_nth_label = 10
        fig_ind = 0


        # distance plot ##########################################################################
        plt.figure(fig_ind)
        fig_ind += 1
        success_count = {}
        total_count = {}
        pcnt = {}
        leg_hands = []
        leg_str = []
        all_dist_err = []

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            total_count[cl] = np.zeros((nx))
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            for err_log in err_log_list:
                for thresh_ind, (dist_thresh, ang_thresh) in enumerate(thresh_list):
                    for (x_err, y_err, z_err, ang_err) in zip(err_log['x_err'], err_log['y_err'], err_log['z_err'], err_log['ang_err']):
                        dist_err = la.norm([x_err, y_err, z_err])
                        all_dist_err.append(dist_err)
                        total_count[cl][thresh_ind] += 1
                        if np.abs(dist_err) < dist_thresh:
                            success_count[cl][thresh_ind] += 1
            pcnt[cl] = np.array([s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
            leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=1)[0])
            leg_str.append(cl)
        ax = plt.gca()
        x_tick_strs = ["{:.2f} m".format(d) for i, (d, a) in enumerate(thresh_list) if i % show_every_nth_label == 0]
        plt.xticks(range(0, len(thresh_list), show_every_nth_label), x_tick_strs, size='small')
        ax.set_xlabel("error metric")
        ax.set_ylabel("% within translation threshold")
        ax.set_title("Translation Error")
        plt.legend(leg_hands, leg_str)
        plt.show(block=False)

        print("Avg translation error: "+str(np.mean(all_dist_err))+" m")
        ##########################################################################


        # angle plot ##########################################################################
        plt.figure(fig_ind)
        fig_ind += 1
        success_count = {}
        total_count = {}
        pcnt = {}
        leg_hands = []
        leg_str = []

        all_ang_err = []

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            total_count[cl] = np.zeros((nx))
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            for err_log in err_log_list:
                for thresh_ind, (dist_thresh, ang_thresh) in enumerate(thresh_list):
                    for (x_err, y_err, z_err, ang_err) in zip(err_log['x_err'], err_log['y_err'], err_log['z_err'], err_log['ang_err']):
                        # dist_err = la.norm([x_err, y_err, z_err])
                        all_ang_err.append(ang_err)
                        total_count[cl][thresh_ind] += 1
                        if np.abs(ang_err) < ang_thresh:
                            success_count[cl][thresh_ind] += 1
            pcnt[cl] = np.array([s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
            leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=1)[0])
            leg_str.append(cl)
        ax = plt.gca()
        # plt.xticks(range(nx), ["({:.2f} m, {:.1f} deg)".format(d, a) for (d, a) in thresh_list], size='small')
        x_tick_strs = ["{:d} deg".format(np.round(a).astype(int)) for i, (d, a) in enumerate(thresh_list) if i % show_every_nth_label == 0]
        plt.xticks(range(0, len(thresh_list), show_every_nth_label), x_tick_strs, size='small')
        ax.set_xlabel("error metric")
        ax.set_ylabel("% within angle threshold")
        ax.set_title("Rotation Error")
        plt.legend(leg_hands, leg_str)
        plt.show(block=False)
        
        print("Avg rotation error: "+str(np.mean(all_ang_err))+" deg")

        ##########################################################################


        # in plane translation err plot ##########################################################################
        plt.figure(fig_ind)
        fig_ind += 1
        success_count = {}
        total_count = {}
        pcnt = {}
        leg_hands = []
        leg_str = []

        all_ip_trans_err = []

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            total_count[cl] = np.zeros((nx))
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            for err_log in err_log_list:
                for thresh_ind, (dist_thresh, ang_thresh) in enumerate(thresh_list):
                    for (x_err, y_err, z_err, ang_err) in zip(err_log['x_err'], err_log['y_err'], err_log['z_err'], err_log['ang_err']):
                        dist_err = la.norm([y_err, z_err])
                        all_ip_trans_err.append(dist_err)
                        total_count[cl][thresh_ind] += 1
                        if np.abs(dist_err) < dist_thresh:
                            success_count[cl][thresh_ind] += 1
            pcnt[cl] = np.array([s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
            leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=1)[0])
            leg_str.append(cl)
        ax = plt.gca()
        x_tick_strs = ["{:.2f} m".format(d) for i, (d, a) in enumerate(thresh_list) if i % show_every_nth_label == 0]
        plt.xticks(range(0, len(thresh_list), show_every_nth_label), x_tick_strs, size='small')
        ax.set_xlabel("error metric")
        ax.set_ylabel("% within distance threshold")
        ax.set_title("In-Plane Translation Error")
        plt.legend(leg_hands, leg_str)
        plt.show(block=False)

        print("Avg in-plane translation error: "+str(np.mean(all_ip_trans_err))+" m")

        ##########################################################################


        # depth translation err plot ##########################################################################
        plt.figure(fig_ind)
        fig_ind += 1
        success_count = {}
        total_count = {}
        pcnt = {}
        leg_hands = []
        leg_str = []

        all_depth_err = []

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            total_count[cl] = np.zeros((nx))
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            for err_log in err_log_list:
                for thresh_ind, (dist_thresh, ang_thresh) in enumerate(thresh_list):
                    for (x_err, y_err, z_err, ang_err) in zip(err_log['x_err'], err_log['y_err'], err_log['z_err'], err_log['ang_err']):
                        dist_err = la.norm([x_err])
                        all_depth_err.append(dist_err)
                        total_count[cl][thresh_ind] += 1
                        if np.abs(dist_err) < dist_thresh:
                            success_count[cl][thresh_ind] += 1
            pcnt[cl] = np.array([s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
            leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=1)[0])
            leg_str.append(cl)
        ax = plt.gca()
        x_tick_strs = ["{:.2f} m".format(d) for i, (d, a) in enumerate(thresh_list) if i % show_every_nth_label == 0]
        plt.xticks(range(0, len(thresh_list), show_every_nth_label), x_tick_strs, size='small')
        ax.set_xlabel("error metric")
        ax.set_ylabel("% within distance threshold")
        ax.set_title("Depth Translation Error")
        plt.legend(leg_hands, leg_str)
        plt.show(block=False)

        print("Avg depth translation error: "+str(np.mean(all_depth_err))+" m")

        ##########################################################################



if __name__ == '__main__':
    try:
        # tell user what inputs are required if missing
        if len(sys.argv) < 2:
            raise RuntimeError("must input <base directory> followed by one or more <class labels> that match folder names in the base directory")
        
        # read in args
        my_base_directory = sys.argv[1]
        my_class_labels = sys.argv[2:]

        # add trailing backslash if missing so appended files work consistantly
        if not my_base_directory[-1] == '/':
            my_base_directory += '/'

        # make sure base dir exists!
        if not os.path.isdir(my_base_directory):
            raise RuntimeError("{} not a valid directory!".format(my_base_directory))
        
        # make sure sub-folders exist!
        for cl in my_class_labels:
            if not os.path.isdir(my_base_directory + cl):
                raise RuntimeError("{} not a valid directory!".format(my_base_directory + cl))

        np.set_printoptions(linewidth=160, suppress=True)  # format numpy so printing matrices is more clear
        program = MultiObjectPlotGenerator(base_directory=my_base_directory, class_labels=my_class_labels)
        input("\nPress enter to close program\n")
        
    except:
        import traceback
        traceback.print_exc()

