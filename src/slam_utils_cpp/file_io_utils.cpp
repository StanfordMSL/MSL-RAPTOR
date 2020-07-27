#include "file_io_utils.h"

using namespace std;
namespace rslam_utils {

  void load_rosbag(vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, tuple<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data, // set<double> &times, map<string, object_est_gt_data_vec_t> &obj_data, 
                    string rosbag_fn, string ego_ns, map<string, obj_param_t> obj_param_map, double dt_thresh) {
    // OUTPUT:  set<double> times; map<string, object_est_gt_data_vec_t> obj_data  [string is name, object_est_gt_data_vec_t is vector of <time, pose gt, pose est>]
    ROS_INFO("loading rosbag: %s", rosbag_fn.c_str());

    cout << "WARNING!!! USING HARDCODED Class-to-object_str map! This is for debug only!" << endl;
    map<string, string> class_to_ado_long_name = {
      {"bowl", "bowl_white_small_norm"},
      {"camera", "camera_canon_len_norm"},
      {"can", "can_arizona_tea_norm"}, 
      {"laptop", "laptop_air_xin_norm"},
      {"cup", "mug_daniel_norm"}
    };

    string gt_pose_topic  = "/mavros/vision_pose/pose";
    string est_pose_topic  = "/mavros/local_position/pose";
    string ego_gt_topic_str = "/" + ego_ns + gt_pose_topic;
    string ego_est_topic_str = "/" + ego_ns + est_pose_topic;
    string raptor_topic_str = "/" + ego_ns + "/msl_raptor_state";
    vector<string> ado_long_names, ado_gt_topic_strs;
    map<string, string> ado_topic_to_name;
    for (const auto & key_val : obj_param_map) {
        string ado_long_name = key_val.first;
        ado_long_names.push_back(ado_long_name);
        ado_gt_topic_strs.push_back("/" + ado_long_name + gt_pose_topic);
        ado_topic_to_name["/" + ado_long_name + gt_pose_topic] = ado_long_name;
    }

    object_data_vec_t ego_data_gt, ego_data_est;   
    map<string, object_data_vec_t> ado_data_gt, ado_data_est;

    rosbag::Bag bag;
    bag.open(rosbag_fn, rosbag::bagmode::Read);
    int num_msg_total = 0;
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
            }
        }
        else if (m.getTopic() == ego_est_topic_str || ("/" + m.getTopic() == ego_est_topic_str)) {
            geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
            if (geo_msg != nullptr) {
                ego_data_est.push_back(make_tuple(time, ros_geo_pose_to_gtsam(geo_msg->pose)));
            }
        }
        else if (m.getTopic() == raptor_topic_str || ("/" + m.getTopic() == raptor_topic_str)) {
          msl_raptor::TrackedObjects::ConstPtr raptor_msg = m.instantiate<msl_raptor::TrackedObjects>();
          if (raptor_msg != nullptr) {
            for (const auto & tracked_obj : raptor_msg->tracked_objects){
              // ROS_INFO("%s", tracked_obj.class_str.c_str());
              geometry_msgs::PoseStamped tmp_pose_stamped = tracked_obj.pose;
              geometry_msgs::Pose tmp_pose = tmp_pose_stamped.pose;
              gtsam::Pose3 tmp = ros_geo_pose_to_gtsam(tmp_pose);
              ado_data_est[class_to_ado_long_name[tracked_obj.class_str]].push_back(make_tuple(time, ros_geo_pose_to_gtsam(tracked_obj.pose.pose))); // tracked_obj.pose is a posestamped, which has a field called pose
            }
          }
        }
        else {
          for (const auto &  topic_str : ado_gt_topic_strs) {
            if (m.getTopic() == topic_str || ("/" + m.getTopic() == topic_str)) {
              geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
              if (geo_msg != nullptr) {
                  ado_data_gt[ado_topic_to_name[topic_str]].push_back(make_tuple(time, ros_geo_pose_to_gtsam(geo_msg->pose)));
                  break;
              }
            }
          }
      }
      num_msg_total++;
    }
    cout << "num messages = " << num_msg_total << endl;
    bag.close();

    map<string, object_est_gt_data_vec_t> ado_data;
    for(const auto &key_value_pair : ado_data_gt) {
      string ado_name = key_value_pair.first;
      if(ado_data_est.find(ado_name) == ado_data_est.end()){
        continue; // this means we had optitrack data, but no raptor data for this object
      }
      object_est_gt_data_vec_t ado_data_single;
      sync_est_and_gt(ado_data_est[ado_name], ado_data_gt[ado_name], ado_data_single, obj_param_map[ado_name], dt_thresh);
      ado_data[ado_name] = ado_data_single;
    }
    object_est_gt_data_vec_t ego_data;
    sync_est_and_gt(ego_data_est, ego_data_gt, ego_data, obj_param_t(ego_ns, ego_ns, 1, false, false, false), dt_thresh);
    // vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, tuple<gtsam::Pose3, gtsam::Pose3> > > > raptor_data;
    zip_data_by_ego(raptor_data, ego_data, ado_data, dt_thresh);
    return;
}

void zip_data_by_ego(vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, tuple<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data, 
                      object_est_gt_data_vec_t ego_data, map<string, object_est_gt_data_vec_t> ado_data, double dt_thresh) {
  // Combine all data into a single data structure that can be looped over. Each element simulates a potential "measurement" from msl_raptor

  for (const auto & ego_data_single : ego_data) {
    double time = get<0>(ego_data_single);
    gtsam::Pose3 ego_gt = get<2>(ego_data_single);
    gtsam::Pose3 ego_est = get<3>(ego_data_single);

    map<string, tuple<gtsam::Pose3, gtsam::Pose3> > measurements;
    bool b_meas_found = false;

    for (const auto & key_val : ado_data) {
      string ado_name = key_val.first;
      object_est_gt_data_vec_t ado_data_one_obj = key_val.second;
      for (const auto & ado_data_single : ado_data_one_obj) {
        double t_est = get<0>(ado_data_single);
        if(dt_thresh > abs(time - t_est)) {
          gtsam::Pose3 ado_gt = get<2>(ado_data_single);
          gtsam::Pose3 ado_est = get<3>(ado_data_single);
          measurements[ado_name] = make_tuple(ado_gt, ado_est);
          b_meas_found = true;
          break;
        }
      }
    }
    if (b_meas_found) {
      raptor_data.emplace_back(time, ego_gt, ego_est, measurements);
    }
  }
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



void read_gt_datafiles(const string fn, map<double, pair<gtsam::Pose3, gtsam::Pose3> >& time_tf_w_ego_map, set<double> &times) {
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

  bool b_verbose = true;
  
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

void write_batch_slam_inputs_csv(string fn, vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, tuple<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data, 
                                  map<string, obj_param_t> obj_param_map) {
  ofstream myFile(fn);
  int t_ind = 0;
  for (const auto & raptor_step : raptor_data ) {
    double time = get<0>(raptor_step);
    gtsam::Pose3 tf_w_ego_gt = get<1>(raptor_step);
    gtsam::Pose3 tf_w_ego_est = get<2>(raptor_step);
    map<string, tuple<gtsam::Pose3, gtsam::Pose3> > measurements = get<3>(raptor_step);

    int ego_pose_index = 1 + t_ind;
    gtsam::Symbol ego_sym = gtsam::Symbol('x', ego_pose_index);
    myFile << time << ", " << ego_sym << ", " << pose_to_string_line(tf_w_ego_gt) << ", " 
                                              << pose_to_string_line(tf_w_ego_est) << ", " 
                                              << pose_to_string_line(tf_w_ego_est) << "\n";

    for (const auto & single_ado_meas : measurements) {
      string ado_name = single_ado_meas.first;
      gtsam::Pose3 tf_w_ado_gt = get<0>(single_ado_meas.second);
      gtsam::Pose3 tf_w_ado_est = get<1>(single_ado_meas.second);
      gtsam::Symbol ado_sym = gtsam::Symbol('l', obj_param_map[ado_name].obj_id);

      myFile << -1 << ", " << ado_sym << ", " << pose_to_string_line(tf_w_ado_gt) << ", " 
                                              << pose_to_string_line(tf_w_ado_est) << ", " 
                                              << pose_to_string_line(tf_w_ado_est) << "\n";
    }
    t_ind++;
  }
  myFile.close();
}


void write_results_csv2(string fn, vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, tuple<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data, 
                                  map<gtsam::Symbol, gtsam::Pose3> &tf_w_est_preslam_map, 
                                  map<gtsam::Symbol, gtsam::Pose3> &tf_w_est_postslam_map, 
                                  map<gtsam::Symbol, map<gtsam::Symbol, pair<gtsam::Pose3, gtsam::Pose3> > > &tf_ego_ado_maps,
                                  map<string, obj_param_t> &obj_param_map) {
  ofstream myFile(fn);
  int t_ind = 0;
  for (const auto & raptor_step : raptor_data ) {
    int ego_pose_index = 1 + t_ind;
    double time = get<0>(raptor_step);
    gtsam::Symbol ego_sym = gtsam::Symbol('x', ego_pose_index);
    gtsam::Pose3 tf_w_ego_gt = get<1>(raptor_step);
    gtsam::Pose3 tf_w_ego_est_pre = tf_w_est_preslam_map[ego_sym];
    gtsam::Pose3 tf_w_ego_est_post = tf_w_est_postslam_map[ego_sym];

    myFile << time << ", " << ego_sym << ", " << pose_to_string_line(tf_w_ego_gt) << ", " 
                                              << pose_to_string_line(tf_w_ego_est_pre) << ", " 
                                              << pose_to_string_line(tf_w_ego_est_post) << "\n";

    map<gtsam::Symbol, pair<gtsam::Pose3, gtsam::Pose3> > measurements = tf_ego_ado_maps[ego_sym];

    for (const auto & single_ado_meas : measurements) {
      gtsam::Symbol ado_sym = single_ado_meas.first;
      gtsam::Pose3 tf_ego_ado_gt = get<0>(single_ado_meas.second);
      gtsam::Pose3 tf_ego_ado_est = get<1>(single_ado_meas.second);

      gtsam::Pose3 tf_w_ado_gt        = tf_w_ego_gt * tf_ego_ado_gt;
      gtsam::Pose3 tf_w_ado_est_pre   = tf_w_ego_est_pre * tf_ego_ado_est;
      gtsam::Pose3 tf_w_ado_est_post  = tf_w_ego_est_post * tf_ego_ado_est;

      myFile << -1 << ", " << ado_sym << ", " << pose_to_string_line(tf_w_ado_gt) << ", " 
                                              << pose_to_string_line(tf_w_ado_est_pre) << ", " 
                                              << pose_to_string_line(tf_w_ado_est_post) << "\n";
    }
    t_ind++;
  }
  myFile.close();
}

void write_results_csv(string fn, map<gtsam::Symbol, double> ego_time_map, map<gtsam::Symbol, gtsam::Pose3> tf_w_gt_map, map<gtsam::Symbol, gtsam::Pose3> tf_w_est_preslam_map, map<gtsam::Symbol, gtsam::Pose3> tf_w_est_postslam_map, map<gtsam::Symbol, map<gtsam::Symbol, pair<gtsam::Pose3, gtsam::Pose3> > > tf_ego_ado_maps) {
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

void write_all_traj_csv(string fn, map<gtsam::Symbol, map<double, pair<gtsam::Pose3, gtsam::Pose3> > > & all_trajs) {
  ofstream myFile(fn);
  double time = 0;
  for(const auto& sym_map_key_value: all_trajs) {
    gtsam::Symbol ado_sym = gtsam::Symbol(sym_map_key_value.first);
    map<double, pair<gtsam::Pose3, gtsam::Pose3> > ado_traj = sym_map_key_value.second;
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