#include "file_io_utils.h"

namespace rslam_utils {

void load_raptor_output_rosbag(std::string rosbag_fn, std::string ego_ns, std::map<std::string, obj_param_t> obj_param_map) {
    ROS_INFO("loading raptor output rosbag: %s", rosbag_fn.c_str());
    return;
}

void load_gt_rosbag(std::string rosbag_fn, std::string ego_ns, std::map<std::string, obj_param_t> obj_param_map) {
    ROS_INFO("GT - loading rosbag: %s", rosbag_fn.c_str());
    rosbag::Bag bag;
    bag.open(rosbag_fn, rosbag::bagmode::Read);

    // msl_raptor::AngledBbox cust_msg;
    std::string gt_pose_topic  = "/mavros/vision_pose/pose";
    std::string est_pose_topic = "/mavros/local_position/pose";

    std::vector<std::string> ado_names, ado_gt_topic_strs, ado_est_topic_strs;
    std::map<std::string, std::string> ado_topic_to_name;
    for (const auto & key_val : obj_param_map) {
        std::string ado_name = key_val.first;
        ado_names.push_back(ado_name);
        ado_gt_topic_strs.push_back(ado_name + gt_pose_topic);
        ado_est_topic_strs.push_back(ado_name + est_pose_topic);
        ado_topic_to_name[ado_name + gt_pose_topic] = ado_name;
        ado_topic_to_name[ado_name + est_pose_topic] = ado_name;
    }
    std::string ego_gt_topic_str = "/" + ego_ns + gt_pose_topic;
    std::string ego_est_topic_str = "/" + ego_ns + est_pose_topic;

    object_data_vec_t ego_data_gt, ego_data_est;
    std::map<std::string, object_data_vec_t> ado_data_gt, ado_data_est;

    int num_msg_total = 0, num_poses = 0;
    double time = 0.0, time0 = -1, ave_dt = 0, last_time = 0;
    geometry_msgs::PoseStamped::ConstPtr geo_msg = nullptr;
    for(rosbag::MessageInstance const m: rosbag::View(bag)) {
        if(time0 < 0) {
            time0 = m.getTime().toSec();
            time = 0.0;
        }
        else {
            last_time = time;
            time = m.getTime().toSec() - time0;
            ave_dt += time - last_time;
        }

        if (m.getTopic() == ego_gt_topic_str || ("/" + m.getTopic() == ego_gt_topic_str)) {
            geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
            if (geo_msg != nullptr) {
                ego_data_gt.push_back(make_tuple(time, ros_geo_pose_to_gtsam(geo_msg->pose)));
                num_poses++;
            }
        }
        else if (m.getTopic() == ego_est_topic_str || ("/" + m.getTopic() == ego_est_topic_str)) {
            geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
            if (geo_msg != nullptr) {
                ego_data_est.push_back(make_tuple(time, ros_geo_pose_to_gtsam(geo_msg->pose)));
                num_poses++;
            }
        }
        else {
            bool b_found = false;
            for (const auto &  topic_str : ado_gt_topic_strs) {
                if (m.getTopic() == topic_str || ("/" + m.getTopic() == topic_str)) {
                    geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
                    if (geo_msg != nullptr) {
                        ado_data_gt[ado_topic_to_name[topic_str]].push_back(make_tuple(time, ros_geo_pose_to_gtsam(geo_msg->pose)));
                        b_found = true;
                        num_poses++;
                        break;
                    }
                }
            }
            if (!b_found) {
                for (const auto &  topic_str : ado_est_topic_strs) {
                    if (m.getTopic() == topic_str || ("/" + m.getTopic() == topic_str)) {
                        geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
                        if (geo_msg != nullptr) {
                            ado_data_est[ado_topic_to_name[topic_str]].push_back(make_tuple(time, ros_geo_pose_to_gtsam(geo_msg->pose)));
                            b_found = true;
                            num_poses++;
                            break;
                        }
                    }
                }
            }
        }
        num_msg_total++;
    }
    std::cout << "num messages = " << num_msg_total << " (" << num_poses << " poses)" << std::endl;
    bag.close();
    return;
}

gtsam::Pose3 ros_geo_pose_to_gtsam(geometry_msgs::Pose ros_pose) {
  // Convert a ros pose structure to gtsam's Pose3 class
  gtsam::Point3 t = gtsam::Point3(ros_pose.position.x, ros_pose.position.y, ros_pose.position.z);
  gtsam::Rot3 R   = gtsam::Rot3( gtsam::Quaternion(ros_pose.orientation.w, 
                                                   ros_pose.orientation.x, 
                                                   ros_pose.orientation.y, 
                                                   ros_pose.orientation.z) );
  return gtsam::Pose3(R, t);
}



void read_gt_datafiles(const string fn, std::map<double, pair<gtsam::Pose3, gtsam::Pose3> >& time_tf_w_ego_map, set<double> &times) {
  // space deliminated file: Time (s), ado_name, Ado State tf, Ego State tf. (tfs are x/y/z/r11,r12,r13,...,r33)
  ifstream infile(fn);
  string line, dummy_str;
  double time, x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33;
  gtsam::Pose3 pose_tf_w_ado_gt, pose_tf_w_ego_gt;
  while (getline(infile, line)) {
    istringstream iss(line);
    iss >> dummy_str; // this "absorbs" the # at the begining of the line
    iss >> time;
    iss >> dummy_str; // this "absorbs" the ado object's name
    iss >> x;
    iss >> y;
    iss >> z;
    iss >> r11;
    iss >> r12;
    iss >> r13;
    iss >> r21;
    iss >> r22;
    iss >> r23;
    iss >> r31;
    iss >> r32;
    iss >> r33;
    pose_tf_w_ado_gt = gtsam::Pose3(gtsam::Rot3(r11, r12, r13, r21, r22, r23, r31, r32, r33), gtsam::Point3(x, y, z));
    iss >> x;
    iss >> y;
    iss >> z;
    iss >> r11;
    iss >> r12;
    iss >> r13;
    iss >> r21;
    iss >> r22;
    iss >> r23;
    iss >> r31;
    iss >> r32;
    iss >> r33;
    pose_tf_w_ego_gt = gtsam::Pose3(gtsam::Rot3(r11, r12, r13, r21, r22, r23, r31, r32, r33), gtsam::Point3(x, y, z));
    // cout << pose_tf_w_ado_gt << endl;
    times.insert(time);
    time_tf_w_ego_map[time] = make_pair(pose_tf_w_ego_gt, pose_tf_w_ado_gt);
    // obj_data.push_back(make_tuple(time, pose_tf_w_ego_gt));
    // obj_data.push_back(make_tuple(time, pose_tf_w_ego_gt.inverse() * pose_tf_w_ado_gt));
    // obj_data.push_back(make_tuple(time, rslam_utils::remove_yaw(pose)));
  }
  return;
}

void read_data_from_one_log(const string fn, object_data_vec_t& obj_data, set<double> &times) {
  // log file header: Time (s), Ado State GT, Ego State GT, 3D Corner GT (X|Y|Z), Corner 2D Projections GT (r|c), Angled BB (r|c|w|h|ang_deg), Image Segmentation Mode
  // note: the states are position (3), lin vel (3), quat wxyz (4), ang vel (3) (space deliminated)
  ifstream infile(fn);
  string line, s;
  double time, x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz;
  gtsam::Pose3 pose;
  getline(infile, line); // skip header of file
  while (getline(infile, line)) {
    istringstream iss(line);
    iss >> time;
    iss >> x;
    iss >> y;
    iss >> z;
    iss >> vx;
    iss >> vy;
    iss >> vz;
    iss >> qw;
    iss >> qx;
    iss >> qy;
    iss >> qz;
    iss >> wx;
    iss >> wy;
    iss >> wz;
    pose = gtsam::Pose3(gtsam::Rot3(gtsam::Quaternion(qw, qx, qy, qz)), gtsam::Point3(x, y, z));
    // NOTE: this conversion from state to pose works the same as that in our python code (verified with test cases)
    times.insert(time);
    obj_data.push_back(make_tuple(time, pose));
    continue;
  }
}

void sync_est_and_gt(object_data_vec_t data_est, object_data_vec_t data_gt, object_est_gt_data_vec_t& data, obj_param_t params, double dt_thresh) {
  // data.push_back(make_tuple(5, get<1>(data_est[0]), get<1>(data_est[0])));
  // now "sync" the gt and est for each object

  bool b_verbose = false;
  
  double t_gt, t_est;
  uint next_est_time_ind = 0;
  for (uint i = 0; i < data_gt.size(); i++) {
    t_gt = get<0>(data_gt[i]);
    for (uint j = next_est_time_ind; j < data_est.size(); j++) {
      t_est = get<0>(data_est[j]);
      if(dt_thresh > abs(t_gt - t_est)) {
        data.push_back(make_tuple((t_gt + t_est)/2, params.obj_id, get<1>(data_gt[i]), get<1>(data_est[j])));
        if (params.obj_id != 1) { // no pose data for ego robot, all zeros
          double t_diff, rot_diff; 
          rslam_utils::calc_pose_delta(get<1>(data_gt[i]).inverse(), get<1>(data_est[j]), &t_diff, &rot_diff, true);

          if (b_verbose) {
            cout << "\n-------------------------------------------------------------" << endl;
            cout << "a) time = " << t_est << ". id = " << params.obj_id << ".  gt / est diff:  t_delta = " << t_diff << ", r_delta = " << rot_diff << " deg" << endl;
          }
          
          double t_diff2, rot_diff2;
          if (!params.b_rm_roll && !params.b_rm_pitch && !params.b_rm_yaw) {
            rslam_utils::calc_pose_delta(get<1>(data_gt[i]).inverse(), get<1>(data_est[j]), &t_diff2, &rot_diff2, true);
            if (b_verbose) {cout << "b) \t\t\t   not symetric" << endl;}
          }
          else {
            if (params.b_rm_roll) {
              runtime_error("Need to implement remove_roll()!");
            }
            if (params.b_rm_pitch) {
              runtime_error("Need to implement remove_pitch()!");
            }
            if (params.b_rm_yaw) {
              gtsam::Pose3 data_gt_no_yaw = rslam_utils::remove_yaw(get<1>(data_gt[i]));
              gtsam::Pose3 data_est_no_yaw = rslam_utils::remove_yaw(get<1>(data_est[i]));
              rslam_utils::calc_pose_delta(data_gt_no_yaw.inverse(), data_est_no_yaw, &t_diff2, &rot_diff2, true);
              if (b_verbose) {cout << "c) \t\t\t   w/o yaw:  t_delta2 = " << t_diff2 << ", r_delta2 = " << rot_diff2 << " deg" << endl;}
              rslam_utils::calc_pose_delta(data_gt_no_yaw.inverse(), get<1>(data_gt[j]), &t_diff2, &rot_diff2, true);
              if (b_verbose) {cout << "d) \t\t\t   w/ vs. w/o yaw [gt]:   t_diff = " << t_diff2 << ", r_diff = " << rot_diff2 << " deg" << endl;}
              rslam_utils::calc_pose_delta(data_est_no_yaw.inverse(), get<1>(data_est[j]), &t_diff2, &rot_diff2, true);
              if (b_verbose) {cout << "e) \t\t\t   w/ vs. w/o yaw [est]:  t_diff = " << t_diff2 << ", r_diff = " << rot_diff2 << " deg" << endl;}

              if (b_verbose && (t_est > 31.7 && params.obj_id == 2 && t_est < 31.9) ) {
                cout << "f) gt yaw: "<< get<1>(data_gt[i]) << endl;              
                cout << "g) gt no yaw: " << data_gt_no_yaw << endl;
                cout << endl;  
              }
            }
          }
          
        }
        next_est_time_ind = j + 1;
        break;
      }
    }
  }
  cout << endl;
}

void write_results_csv(string fn, std::map<gtsam::Symbol, double> ego_time_map, std::map<gtsam::Symbol, gtsam::Pose3> tf_w_gt_map, std::map<gtsam::Symbol, gtsam::Pose3> tf_w_est_preslam_map, std::map<gtsam::Symbol, gtsam::Pose3> tf_w_est_postslam_map, std::map<gtsam::Symbol, std::map<gtsam::Symbol, pair<gtsam::Pose3, gtsam::Pose3> > > tf_ego_ado_maps) {
  ofstream myFile(fn);
  double time = 0;
  for(const auto& key_value: tf_w_gt_map) {
    gtsam::Symbol ego_sym = gtsam::Symbol(key_value.first);
    gtsam::Pose3 tf_w_ego_gt = tf_w_gt_map[ego_sym];
    gtsam::Pose3 tf_w_ego_est_preslam = tf_w_est_preslam_map[ego_sym];
    gtsam::Pose3 tf_w_ego_est_postslam = tf_w_est_postslam_map[ego_sym];

    if(ego_sym.chr() == 'l' || ego_sym.chr() == 'L'){
      time = 0.0;
      myFile << time << ", " << ego_sym << ", " << pose_to_string_line(tf_w_ego_gt) << ", " 
                                                << pose_to_string_line(tf_w_ego_est_preslam) << ", " 
                                                << pose_to_string_line(tf_w_ego_est_postslam) << "\n";
    }
    else {
      time = ego_time_map[ego_sym];
      myFile << time << ", " << ego_sym << ", " << pose_to_string_line(tf_w_ego_gt) << ", " 
                                                << pose_to_string_line(tf_w_ego_est_preslam) << ", " 
                                                << pose_to_string_line(tf_w_ego_est_postslam) << "\n";

      for (const auto & key_value : tf_ego_ado_maps[ego_sym]) {
        gtsam::Symbol ado_sym = key_value.first;
        pair<gtsam::Pose3, gtsam::Pose3> pose_gt_est_pair = key_value.second;
        gtsam::Pose3 tf_ego_ado_gt  = pose_gt_est_pair.first;
        gtsam::Pose3 tf_ego_ado_est = pose_gt_est_pair.second;
        gtsam::Pose3 tf_w_ado_gt        = tf_w_ego_gt * tf_ego_ado_gt;
        gtsam::Pose3 tf_w_ado_est_pre   = tf_w_ego_est_preslam * tf_ego_ado_est;
        gtsam::Pose3 tf_w_ado_est_post  = tf_w_ego_est_postslam * tf_ego_ado_est;
        myFile << -1 << ", " << ado_sym << ", " << pose_to_string_line(tf_w_ado_gt) << ", " 
                                                << pose_to_string_line(tf_w_ado_est_pre) << ", " 
                                                << pose_to_string_line(tf_w_ado_est_post) << "\n";
      }
    }
  }
  myFile.close();
}

void write_all_traj_csv(string fn, std::map<gtsam::Symbol, std::map<double, pair<gtsam::Pose3, gtsam::Pose3> > > & all_trajs) {
  ofstream myFile(fn);
  double time = 0;
  for(const auto& sym_map_key_value: all_trajs) {
    gtsam::Symbol ado_sym = gtsam::Symbol(sym_map_key_value.first);
    std::map<double, pair<gtsam::Pose3, gtsam::Pose3> > ado_traj = sym_map_key_value.second;
    for(const auto& time_poses_key_value: ado_traj) {
      double time = time_poses_key_value.first;
      pair<gtsam::Pose3, gtsam::Pose3> pose_gt_est_pair = time_poses_key_value.second;
      gtsam::Pose3 tf_w_ego_gt  = pose_gt_est_pair.first;
      gtsam::Pose3 tf_w_ego_est = pose_gt_est_pair.second;
      myFile << ado_sym << ", " << time << ", " << pose_to_string_line(tf_w_ego_gt) << ", " << pose_to_string_line(tf_w_ego_est) << pose_to_string_line((tf_w_ego_gt.inverse()) *tf_w_ego_est ) << "\n";
    }
  }
  myFile.close();
}

string pose_to_string_line(gtsam::Pose3 p) {
  string s;
  gtsam::Point3 t = p.translation();
  gtsam::Rot3 R = p.rotation();
  gtsam::Matrix3 M = R.matrix();
  s = to_string(t.x()) + ", ";
  s += to_string(t.y()) + ", ";
  s += to_string(t.z()) + ", ";
  s += to_string(M(0,0)) + ", ";
  s += to_string(M(0,1)) + ", ";
  s += to_string(M(0,2)) + ", ";
  s += to_string(M(1,0)) + ", ";
  s += to_string(M(1,1)) + ", ";
  s += to_string(M(1,2)) + ", ";
  s += to_string(M(2,0)) + ", ";
  s += to_string(M(2,1)) + ", ";
  s += to_string(M(2,2));
  return s;
}


}