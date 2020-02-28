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
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
# Utils
sys.path.append('/root/msl_raptor_ws/src/msl_raptor/src/utils_msl_raptor')
from raptor_logger import *
from itertools import chain
from ssp_utils import makedirs

class MultiObjectPlotGenerator:

    def __init__(self, base_directory, class_labels):

        # PLOT OPTIONS ###################################
        b_nocs = False
        b_save_figs = True
        self.b_show_figs = True
        img_path = '/mounted_folder/saved_figs/'
        makedirs(img_path)
        color_strs = ['r', 'b', 'm', 'k', 'c', 'g']
        fontsize = 40
        tick_fontsize = 34
        legend_fontsize = 26
        linewidth = 3
        text_weight = 'normal'  # 'normal' or 'bold'
        # font_path = '/usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono-Bold.ttf'  # this one has bold
        # font_path = '/usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/fonts/ttf/STIXGeneral.ttf'  # this one does not...
        self.plot_scale = 1.8 # make plots bigger so there is space for legend
        b_capitalize_names = True
        b_show_dots = False
        b_plot_titles = False
        # leg_font_props = mfm.FontProperties(fname=font_path, weight=text_weight, size=fontsize-2)
        perp_symbol = '$^+$'#u'\u27c2'
        prrl_symbol = '$^=$'#u'\u2225' # prrl_symbol = '||'  

        # font = {'family' : 'serif',
        #         'weight' : text_weight,
        #         'size'   : fontsize}
        # matplotlib.rc('font', **font)
        plt.rc('font', family='serif', size=fontsize, weight=text_weight)
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')

        if b_nocs:
            nx = 20  # number of ticks on x axis
            show_every_nth_label = 5
            d_max = 0.2  # max trans err thresh
            d_unitless_max = 0.1  # max trans error / distance thresh
            a_max = 60  # max angle thresh
            p_max = 50  # max pixel thresh
        else:
            nx = 20  # number of ticks on x axis
            show_every_nth_label = 5
            d_max = 3
            d_unitless_max = 1.5
            a_max = 30
            p_max = 50

        
        major_ticks_x = np.arange(0, nx + .01, show_every_nth_label)
        minor_ticks_x = np.arange(0, nx + .01, 1)
        x_dist_labels_to_show = np.round((np.linspace(0, d_max, show_every_nth_label) * 100)) / 100
        x_dist_unitless_labels_to_show = np.linspace(0, d_unitless_max, show_every_nth_label)
        x_ang_labels_to_show = np.linspace(0, a_max, show_every_nth_label).astype(int)
        x_pix_labels_to_show = np.linspace(0, p_max, show_every_nth_label).astype(int)
        dist_thresh = np.linspace(0, d_max, nx)
        dist_unitless_thresh = np.linspace(0, d_unitless_max, nx)
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
        all_dist_err = {}

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            total_count[cl] = np.ones((nx))*self.eps
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            all_dist_err[cl] = []
            for err_log in err_log_list:
                for thresh_ind, (dist_thresh, ang_thresh, pix_thresh) in enumerate(thresh_list):
                    for (x_err, y_err, z_err, ang_err) in zip(err_log['x_err'], err_log['y_err'], err_log['z_err'], err_log['ang_err']):
                        dist_err = la.norm([x_err, y_err, z_err])
                        all_dist_err[cl].append(dist_err)
                        total_count[cl][thresh_ind] += 1
                        if np.abs(dist_err) < dist_thresh:
                            success_count[cl][thresh_ind] += 1
            pcnt[cl] = np.array([100*s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            if b_show_dots:
                plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
            leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=linewidth)[0])
            leg_str.append(cl)
        ax = plt.gca()
        self.adjust_axes(ax, major_ticks_x, minor_ticks_x, x_dist_labels_to_show)

        ax.set_xlabel("translation error threshold (m)", size=tick_fontsize, weight=text_weight)
        ax.set_ylabel("correct estimates in %", size=tick_fontsize, weight=text_weight)
        if b_plot_titles:
            ax.set_title("Translation Error")
        plt.legend(leg_hands, leg_str, loc=0, prop={'size': legend_fontsize, 'weight': text_weight})
        if self.b_show_figs:
            plt.show(block=False)
        if b_save_figs:
            plt.savefig(img_path + '/s_curve_trans.png', bbox_inches='tight')

        for k in all_dist_err.keys():
            print(k+ " avg translation error: "+str(np.mean(all_dist_err[k]))+" m")


        print("Total avg translation error: "+str(np.mean(list(chain.from_iterable(list(all_dist_err.values())))))+" m\n")
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
        all_ang_err = {}

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            total_count[cl] = np.ones((nx))*self.eps
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            all_ang_err[cl] = []
            for err_log in err_log_list:
                for thresh_ind, (dist_thresh, ang_thresh, pix_thresh) in enumerate(thresh_list):
                    for (x_err, y_err, z_err, ang_err) in zip(err_log['x_err'], err_log['y_err'], err_log['z_err'], err_log['ang_err']):
                        all_ang_err[cl].append(ang_err)
                        total_count[cl][thresh_ind] += 1
                        if np.abs(ang_err) < ang_thresh:
                            success_count[cl][thresh_ind] += 1
            pcnt[cl] = np.array([100*s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            if b_show_dots:
                plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
            leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=linewidth)[0])
            leg_str.append(cl)
        ax = plt.gca()
        self.adjust_axes(ax, major_ticks_x, minor_ticks_x, x_ang_labels_to_show)
        ax.set_xlabel("angle threshold (deg)", fontsize=tick_fontsize, weight=text_weight)
        ax.set_ylabel("correct estimates in %", fontsize=tick_fontsize, weight=text_weight)
        if b_plot_titles:
            ax.set_title("Rotation Error")
        plt.legend(leg_hands, leg_str, loc=0, prop={'size': legend_fontsize, 'weight': text_weight})
        
        if self.b_show_figs:
            plt.show(block=False)
        if b_save_figs:
            plt.savefig(img_path + '/s_curve_rot.png', bbox_inches='tight')
        
        for k in all_ang_err.keys():
            print(k+ " avg rotation error: "+str(np.mean(all_ang_err[k]))+" deg")

        print("Avg rotation error: "+str(np.mean(list(chain.from_iterable(list(all_ang_err.values())))))+" deg\n")

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
        all_ip_trans_err = {}
        all_depth_err = {}

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            success_count2[cl] = np.zeros((nx))
            total_count[cl] = np.ones((nx))*self.eps
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            all_ip_trans_err[cl] = []
            all_depth_err[cl] = []
            for err_log in err_log_list:
                for thresh_ind, (dist_thresh, ang_thresh, pix_thresh) in enumerate(thresh_list):
                    for (x_err, y_err, z_err, ang_err) in zip(err_log['x_err'], err_log['y_err'], err_log['z_err'], err_log['ang_err']):
                        dist_err_inplane = la.norm([y_err, z_err])
                        dist_err_depth = la.norm([x_err])
                        all_ip_trans_err[cl].append(dist_err_inplane)
                        all_depth_err[cl].append(dist_err_depth)
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
            leg_str.append(perp_symbol + cl)
            leg_hands.append(plt.plot(range(nx), pcnt2[cl], color_strs[i] + '--', linewidth=linewidth)[0])
            leg_str.append(prrl_symbol + cl)
        ax = plt.gca()
        self.adjust_axes(ax, major_ticks_x, minor_ticks_x, x_dist_labels_to_show)
        ax.set_xlabel("translation error threshold (m)", fontsize=tick_fontsize, weight=text_weight)
        ax.set_ylabel("correct estimates in %", fontsize=tick_fontsize, weight=text_weight)
        if b_plot_titles:
            ax.set_title("Translation Error (In-Plane vs. Depth)")
        plt.legend(leg_hands, leg_str, loc=0, prop={'size': legend_fontsize, 'weight': text_weight})
        if self.b_show_figs:
            plt.show(block=False)
        if b_save_figs:
            plt.savefig(img_path + '/s_curve_trans_inplanedepth.png', bbox_inches='tight')

        for k in all_depth_err.keys():
            print(k+ " avg depth translation error: "+str(np.mean(all_depth_err[k]))+" m")

        print("Avg depth translation error: "+str(np.mean(list(chain.from_iterable(list(all_depth_err.values())))))+" m\n")

        for k in all_ip_trans_err.keys():
            print(k+ " avg depth in plane translation error: "+str(np.mean(all_ip_trans_err[k]))+" m")
       
        print("Avg depth translation error: "+str(np.mean(list(chain.from_iterable(list(all_ip_trans_err.values())))))+" m\n")


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
        all_trans_err_per_depth = defaultdict(list)

        for i, cl in enumerate(err_log_dict):
            success_count[cl] = np.zeros((nx))
            total_count[cl] = np.ones((nx))*self.eps
            pcnt[cl] = np.zeros((nx))
            err_log_list = err_log_dict[cl]
            for err_log in err_log_list:
                for thresh_ind, (unitless_thresh) in enumerate(dist_unitless_thresh):
                    for (x_err, y_err, z_err, ang_err, meas_dist) in zip(err_log['x_err'], err_log['y_err'], err_log['z_err'], err_log['ang_err'], err_log["measurement_dist"]):
                        dist_err = la.norm([x_err, y_err, z_err])
                        total_count[cl][thresh_ind] += 1
                        t_err_per_depth = np.abs(dist_err)/meas_dist
                        all_trans_err_per_depth[cl].append(t_err_per_depth)
                        if t_err_per_depth < unitless_thresh:
                            success_count[cl][thresh_ind] += 1
            pcnt[cl] = np.array([100*s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
            if b_show_dots:
                plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
            leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=linewidth)[0])
            leg_str.append(cl)
        ax = plt.gca()
        self.adjust_axes(ax, major_ticks_x, minor_ticks_x, x_dist_labels_to_show)
        ax.set_xlabel("translation error / distance threshold (m/m)", fontsize=tick_fontsize, weight=text_weight)
        ax.set_ylabel("correct estimates in %", fontsize=tick_fontsize, weight=text_weight)
        if b_plot_titles:
            ax.set_title("Translation Error / Distance to Object")
        plt.legend(leg_hands, leg_str, loc=0, prop={'size': legend_fontsize, 'weight': text_weight})
        if self.b_show_figs:
            plt.show(block=False)
        if b_save_figs:
            plt.savefig(img_path + '/s_curve_trans_per_depth.png', bbox_inches='tight')
        
        for k in all_trans_err_per_depth.keys():
            print(k+ " avg translation errpr / depth: "+str(np.mean(all_trans_err_per_depth[k]))+" m")
        # med_t_err_msl = np.median(all_dist_err["mslraptor".upper()])
        # med_t_err_ssp = np.median(all_dist_err["ssp".upper()])
        # med_r_err_msl = np.median(all_ang_err["mslraptor".upper()])
        # med_r_err_ssp = np.median(all_ang_err["ssp".upper()])
        # print("Median dist err = {} m for mslraptor and {} m for ssp\nMedian rotation error = {} deg for mslraptor and {} deg for ssp".format(med_t_err_msl, med_t_err_ssp, med_r_err_msl, med_r_err_ssp))
        # pdb.set_trace()
        ##########################################################################
        
        # # pixel to target plot ###################### error####################################################
        # plt.figure(fig_ind)
        # fig_ind += 1
        # self.adjust_plot_size()
        # success_count = {}
        # total_count = {}
        # pcnt = {}
        # leg_hands = []
        # leg_str = []

        # for i, cl in enumerate(err_log_dict):
        #     success_count[cl] = np.zeros((nx))
        #     total_count[cl] = np.ones((nx))*self.eps
        #     pcnt[cl] = np.zeros((nx))
        #     err_log_list = err_log_dict[cl]
        #     for err_log in err_log_list:
        #         for thresh_ind, (dist_thresh, ang_thresh, pix_thresh) in enumerate(thresh_list):
        #             for pix_err in err_log["pix_err"]:
        #                 total_count[cl][thresh_ind] += 1
        #                 if np.abs(pix_err) < pix_thresh:
        #                     success_count[cl][thresh_ind] += 1
        #     pcnt[cl] = np.array([100*s/t for s, t in zip(success_count[cl], total_count[cl])]) # elementwise success_count[cl] / total_count[cl]
        #     if b_show_dots:
        #         plt.plot(range(nx), pcnt[cl], color_strs[i] + '.', markersize=4)
        #     leg_hands.append(plt.plot(range(nx), pcnt[cl], color_strs[i] + '-', linewidth=linewidth)[0])
        #     leg_str.append(cl)
        # ax = plt.gca()
        # self.adjust_axes(ax, major_ticks_x, minor_ticks_x, x_dist_labels_to_show)
        # ax.set_xlabel("pixel threshold (pix)", fontsize=tick_fontsize, weight=text_weight)
        # ax.set_ylabel("correct estimates in %", fontsize=tick_fontsize, weight=text_weight)
        # if b_plot_titles:
        #     ax.set_title("Pixel Error")
        # plt.legend(leg_hands, leg_str, loc=0, prop={'size': legend_fontsize, 'weight': text_weight})
        # plt.show(block=False)
        # ##########################################################################

    def adjust_plot_size(self):
        fig_size = plt.gcf().get_size_inches() #Get current size
        sizefactor = self.plot_scale #Set a zoom factor
        # Modify the current size by the factor
        plt.gcf().set_size_inches(sizefactor * fig_size)

    def adjust_axes(self, ax, major_ticks_x, minor_ticks_x, x_dist_labels_to_show):
        ax.set_xticks(major_ticks_x)
        ax.set_xticklabels(x_dist_labels_to_show)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

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
        if program.b_show_figs:
            input("\nPress enter to close program\n")
        
    except:
        import traceback
        traceback.print_exc()

