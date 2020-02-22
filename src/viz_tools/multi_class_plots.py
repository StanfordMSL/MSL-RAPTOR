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
import matplotlib.font_manager as mfm
# Utils
sys.path.append('/root/msl_raptor_ws/src/msl_raptor/src/utils_msl_raptor')
from raptor_logger import *

class MultiObjectPlotGenerator:

    def __init__(self, base_directory, class_labels):

        # PLOT OPTIONS ###################################
        b_nocs = False
        color_strs = ['r', 'b', 'm', 'k', 'c', 'g']
        fontsize = 26
        linewidth = 3
        text_weight = 'bold'  # 'normal' or 'bold'
        # font_path = '/usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono-Bold.ttf'  # this one has bold
        font_path = '/usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/fonts/ttf/STIXGeneral.ttf'  # this one does not...
        self.plot_scale = 1.5 # make plots bigger so there is space for legend
        b_capitalize_names = True
        b_show_dots = False
        b_plot_titles = False
        leg_font_props = mfm.FontProperties(fname=font_path, weight=text_weight, size=fontsize-2)
        perp_symbol = u'\u27c2'
        # prrl_symbol = '||'  
        prrl_symbol = u'\u2225'

        font = {'weight' : text_weight,
                'size'   : fontsize}
        # pdb.set_trace()
        matplotlib.rc('font', **font)

        show_every_nth_label = 5
        if b_nocs:
            nx = 100  # number of ticks on x axis
            d_max = 0.2
            a_max = 60
            p_max = 50
            # x_dist_labels_to_show = np.linspace(0, d_max, 5)
            # x_ang_labels_to_show = np.linspace(0, a_max, 5)
        else:
            nx = 20  # number of ticks on x axis
            d_max = 3
            a_max = 30
            p_max = 50
            # x_dist_labels_to_show = np.linspace(0, d_max, 5)
            # x_ang_labels_to_show = np.linspace(0, a_max, 5)
        dist_thresh = np.linspace(0, d_max, nx)
        ang_thresh = np.linspace(0, a_max, nx)
        pix_thresh = np.linspace(0, p_max, nx)
        ###################################

        self.eps = 1e-5
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
                if b_capitalize_names:
                    err_log_dict[cl.upper()].append(logs)
                else:
                    err_log_dict[cl].append(logs)
        # make plot with varying thresholds
        if len(err_log_dict) > len(color_strs):
            raise RuntimeError("need to add more colors to color string list!! too many classes..")
        thresh_list = list(zip(dist_thresh, ang_thresh, pix_thresh)) # [(m thesh, deg thresh)]
        fig_ind = 0


        # distance plot ##########################################################################
        plt.figure(fig_ind)
        fig_ind += 1
        self.adjust_plot_size()
        success_count = {}
        total_count = {}
        pcnt = {}
        leg_hands = []
        leg_str = []
        all_dist_err = []

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            total_count[cl] = np.ones((nx))*self.eps
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            for err_log in err_log_list:
                for thresh_ind, (dist_thresh, ang_thresh, pix_thresh) in enumerate(thresh_list):
                    for (x_err, y_err, z_err, ang_err) in zip(err_log['x_err'], err_log['y_err'], err_log['z_err'], err_log['ang_err']):
                        dist_err = la.norm([x_err, y_err, z_err])
                        all_dist_err.append(dist_err)
                        total_count[cl][thresh_ind] += 1
                        if np.abs(dist_err) < dist_thresh:
                            success_count[cl][thresh_ind] += 1
            pcnt[cl] = np.array([100*s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            if b_show_dots:
                plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
            leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=linewidth)[0])
            leg_str.append(cl)
        ax = plt.gca()
        x_tick_strs = ["{:.2f} m".format(d) for i, (d, a, p) in enumerate(thresh_list) if i % show_every_nth_label == 0]
        plt.xticks(range(0, len(thresh_list), show_every_nth_label), x_tick_strs, size='small')
        # plt.xticks(x_dist_labels_to_show, ["{:.2} m".format(f) for f in x_dist_labels_to_show], size='small')
        ax.set_xlabel("distance threshold", fontsize=fontsize, weight=text_weight)
        ax.set_ylabel("correct estimates in %", fontsize=fontsize, weight=text_weight)
        if b_plot_titles:
            ax.set_title("Translation Error")
        plt.legend(leg_hands, leg_str, prop=leg_font_props)
        plt.show(block=False)

        print("Avg translation error: "+str(np.mean(all_dist_err))+" m")
        ##########################################################################


        # angle plot ##########################################################################
        plt.figure(fig_ind)
        fig_ind += 1
        self.adjust_plot_size()
        success_count = {}
        total_count = {}
        pcnt = {}
        leg_hands = []
        leg_str = []

        all_ang_err = []

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            total_count[cl] = np.ones((nx))*self.eps
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            for err_log in err_log_list:
                for thresh_ind, (dist_thresh, ang_thresh, pix_thresh) in enumerate(thresh_list):
                    for (x_err, y_err, z_err, ang_err) in zip(err_log['x_err'], err_log['y_err'], err_log['z_err'], err_log['ang_err']):
                        # dist_err = la.norm([x_err, y_err, z_err])
                        all_ang_err.append(ang_err)
                        total_count[cl][thresh_ind] += 1
                        if np.abs(ang_err) < ang_thresh:
                            success_count[cl][thresh_ind] += 1
            pcnt[cl] = np.array([100*s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            if b_show_dots:
                plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
            leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=linewidth)[0])
            leg_str.append(cl)
        ax = plt.gca()
        x_tick_strs = ["{:d} deg".format(np.round(a).astype(int)) for i, (d, a, p) in enumerate(thresh_list) if i % show_every_nth_label == 0]
        plt.xticks(range(0, len(thresh_list), show_every_nth_label), x_tick_strs, size='small')
        # plt.xticks(x_dist_labels_to_show, ["{:d} deg".format(np.round(f).astype(int)) for f in x_ang_labels_to_show], size='small')
        ax.set_xlabel("angle threshold", fontsize=fontsize, weight=text_weight)
        ax.set_ylabel("correct estimates in %", fontsize=fontsize, weight=text_weight)
        if b_plot_titles:
            ax.set_title("Rotation Error")
        plt.legend(leg_hands, leg_str, prop=leg_font_props)
        plt.show(block=False)
        
        print("Avg rotation error: "+str(np.mean(all_ang_err))+" deg")

        ##########################################################################


        # in plane vs depth translation err plot ##########################################################################
        plt.figure(fig_ind)
        fig_ind += 1
        self.adjust_plot_size()
        success_count = {}
        total_count = {}
        pcnt = {}
        success_count2 = {}
        pcnt2 = {}
        leg_hands = []
        leg_str = []

        all_ip_trans_err = []
        all_depth_err = []

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            success_count2[cl] = np.zeros((nx))
            total_count[cl] = np.ones((nx))*self.eps
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            for err_log in err_log_list:
                for thresh_ind, (dist_thresh, ang_thresh, pix_thresh) in enumerate(thresh_list):
                    for (x_err, y_err, z_err, ang_err) in zip(err_log['x_err'], err_log['y_err'], err_log['z_err'], err_log['ang_err']):
                        dist_err_inplane = la.norm([y_err, z_err])
                        dist_err_depth = la.norm([x_err])
                        all_ip_trans_err.append(dist_err_inplane)
                        all_depth_err.append(dist_err_depth)
                        total_count[cl][thresh_ind] += 1
                        if np.abs(dist_err_inplane) < dist_thresh:
                            success_count[cl][thresh_ind] += 1
                        if np.abs(dist_err_depth) < dist_thresh:
                            success_count2[cl][thresh_ind] += 1
            pcnt[cl] = np.array([100*s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            pcnt2[cl] = np.array([100*s/t for s, t in zip(success_count2[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            if b_show_dots:
                plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
                plt.plot(range(nx), pcnt2[cl], color_strs[i] + '.', markersize=4)
            leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=linewidth)[0])
            leg_str.append(perp_symbol + ' ' + cl)
            leg_hands.append(plt.plot(range(nx), pcnt2[cl], color_strs[i] + '--', linewidth=linewidth)[0])
            leg_str.append(prrl_symbol + ' ' + cl)
        ax = plt.gca()
        x_tick_strs = ["{:.2f} m".format(d) for i, (d, a, p) in enumerate(thresh_list) if i % show_every_nth_label == 0]
        plt.xticks(range(0, len(thresh_list), show_every_nth_label), x_tick_strs, size='small')
        # plt.xticks(x_dist_labels_to_show, ["{:.2} m".format(f) for f in x_dist_labels_to_show], size='small')
        ax.set_xlabel("distance threshold", fontsize=fontsize, weight=text_weight)
        ax.set_ylabel("correct estimates in %", fontsize=fontsize, weight=text_weight)
        if b_plot_titles:
            ax.set_title("Translation Error (In-Plane vs. Depth)")
        plt.legend(leg_hands, leg_str, prop=leg_font_props)
        plt.show(block=False)

        print("Avg in-plane translation error: "+str(np.mean(all_ip_trans_err))+" m")
        print("Avg depth translation error: "+str(np.mean(all_depth_err))+" m")

        ##########################################################################
        
        
        # trans error / distance to target plot ##########################################################################
        plt.figure(fig_ind)
        fig_ind += 1
        self.adjust_plot_size()
        success_count = {}
        total_count = {}
        pcnt = {}
        leg_hands = []
        leg_str = []

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            total_count[cl] = np.ones((nx))*self.eps
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            for err_log in err_log_list:
                for thresh_ind, (dist_thresh, ang_thresh, pix_thresh) in enumerate(thresh_list):
                    for (x_err, y_err, z_err, ang_err, meas_dist) in zip(err_log['x_err'], err_log['y_err'], err_log['z_err'], err_log['ang_err'], err_log["measurement_dist"]):
                        dist_err = la.norm([x_err, y_err, z_err])
                        total_count[cl][thresh_ind] += 1
                        if np.abs(dist_err)/meas_dist < dist_thresh:
                            success_count[cl][thresh_ind] += 1
            pcnt[cl] = np.array([100*s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            if b_show_dots:
                plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
            leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=linewidth)[0])
            leg_str.append(cl)
        ax = plt.gca()
        x_tick_strs = ["{:.2f} m".format(d) for i, (d, a, p) in enumerate(thresh_list) if i % show_every_nth_label == 0]
        plt.xticks(range(0, len(thresh_list), show_every_nth_label), x_tick_strs, size='small')
        # plt.xticks(x_dist_labels_to_show, ["{:.2}".format(f) for f in x_dist_labels_to_show], size='small')
        ax.set_xlabel("distance threshold", fontsize=fontsize, weight=text_weight)
        ax.set_ylabel("correct estimates in %", fontsize=fontsize, weight=text_weight)
        if b_plot_titles:
            ax.set_title("Translation Error / Distance to Object")
        plt.legend(leg_hands, leg_str, prop=leg_font_props)
        plt.show(block=False)
        ##########################################################################
        
        # pixel to target plot ##########################################################################
        plt.figure(fig_ind)
        fig_ind += 1
        self.adjust_plot_size()
        success_count = {}
        total_count = {}
        pcnt = {}
        leg_hands = []
        leg_str = []

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            total_count[cl] = np.ones((nx))*self.eps
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            for err_log in err_log_list:
                for thresh_ind, (dist_thresh, ang_thresh, pix_thresh) in enumerate(thresh_list):
                    for pix_err in err_log["pix_err"]:
                        total_count[cl][thresh_ind] += 1
                        if np.abs(pix_err) < pix_thresh:
                            success_count[cl][thresh_ind] += 1
            pcnt[cl] = np.array([100*s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            if b_show_dots:
                plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
            leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=linewidth)[0])
            leg_str.append(cl)
        ax = plt.gca()
        x_tick_strs = ["{:d} pix".format(int(p)) for i, (d, a, p) in enumerate(thresh_list) if i % show_every_nth_label == 0]
        plt.xticks(range(0, len(thresh_list), show_every_nth_label), x_tick_strs, size='small')
        # plt.xticks(x_dist_labels_to_show, ["{:.2}".format(f) for f in x_dist_labels_to_show], size='small')
        ax.set_xlabel("pixel threshold", fontsize=fontsize, weight=text_weight)
        ax.set_ylabel("correct estimates in %", fontsize=fontsize, weight=text_weight)
        if b_plot_titles:
            ax.set_title("Pixel Error")
        plt.legend(leg_hands, leg_str, prop=leg_font_props)
        plt.show(block=False)
        ##########################################################################

    def adjust_plot_size(self):
        fig_size = plt.gcf().get_size_inches() #Get current size
        sizefactor = self.plot_scale #Set a zoom factor
        # Modify the current size by the factor
        plt.gcf().set_size_inches(sizefactor * fig_size)


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

