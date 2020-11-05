#include "msl_raptor_gtsam_utils.h"

//////////////////////////////////////////////////////////
// Other Helper Functions
//////////////////////////////////////////////////////////
void gen_all_fake_trajectories(map<Symbol, map<double, pair<Pose3, Pose3> > > & all_trajs, set<double> times, const object_est_gt_data_vec_t& obj_data, int t_ind_cutoff, double dt_thresh) {
    // build trajectory and get initial measurements of objects
  map<Symbol, Pose3> tf_w_gt_map, tf_w_est_preslam_map; 

  // map<Symbol, map<double, pair<Pose3, Pose3> > > all_trajs;
  // vector<pair<Pose3, Pose3>> tf_w_ego_gt_est_vec;
  int t_ind = 0, ego_pose_index = 0, obj_list_ind = 0;
  for(const auto & ego_time : times) {
    if (t_ind > t_ind_cutoff) {
      break;
    }
    Pose3 tf_w_ego_gt, tf_w_ego_est;
    ego_pose_index = 1 + t_ind;
    Symbol ego_sym = Symbol('x', ego_pose_index);

    while(obj_list_ind < obj_data.size() && abs(get<0>(obj_data[obj_list_ind]) - ego_time) < dt_thresh ) {
      Pose3 tf_ego_ado_est = get<3>(obj_data[obj_list_ind]); // estimated ado pose
      Pose3 tf_ego_ado_gt  = get<2>(obj_data[obj_list_ind]); // current relative gt object pose
      int obj_id = get<1>(obj_data[obj_list_ind]);
      Symbol ado_sym = Symbol('l', obj_id);

      if(t_ind == 0) {
        tf_w_gt_map[ado_sym] = get<2>(obj_data[obj_list_ind]); // tf_w_ado_gt
        tf_w_est_preslam_map[ado_sym] = get<3>(obj_data[obj_list_ind]); // tf_w_ado_est
        tf_w_ego_gt = Pose3();
      }
      else {
        tf_w_ego_gt = tf_w_gt_map[ado_sym] * tf_ego_ado_gt.inverse(); // gt ego pose in world frame
        tf_w_ego_est = tf_w_est_preslam_map[ado_sym] * tf_ego_ado_est.inverse(); // est ego pose in world frame
      }
      all_trajs[ado_sym][ego_time] = make_pair(tf_w_ego_gt, tf_w_ego_est);
      obj_list_ind++;
    }
    t_ind++;
  }
}

void gen_fake_trajectory(vector<pair<Pose3, Pose3>> & tf_w_ego_gt_est_vec, set<double> times, const object_est_gt_data_vec_t& obj_data, int t_ind_cutoff, double dt_thresh) {
    // build trajectory and get initial measurements of objects
  map<Symbol, Pose3> tf_w_gt_map, tf_w_est_preslam_map; 
  // vector<pair<Pose3, Pose3>> tf_w_ego_gt_est_vec;
  int t_ind = 0, ego_pose_index = 0, obj_list_ind = 0;
  for(const auto & ego_time : times) {
    if (t_ind > t_ind_cutoff) {
      break;
    }
    Pose3 tf_w_ego_gt, tf_w_ego_est;
    ego_pose_index = 1 + t_ind;
    Symbol ego_sym = Symbol('x', ego_pose_index);

    while(obj_list_ind < obj_data.size() && abs(get<0>(obj_data[obj_list_ind]) - ego_time) < dt_thresh ) {
      Pose3 tf_ego_ado_est = get<3>(obj_data[obj_list_ind]); // estimated ado pose
      Pose3 tf_ego_ado_gt  = get<2>(obj_data[obj_list_ind]); // current relative gt object pose
      int obj_id = get<1>(obj_data[obj_list_ind]);
      Symbol ado_sym = Symbol('l', obj_id);
      // if(obj_id != 6 && obj_id != 5) {
      if(obj_id == 2 || obj_id == 4) {
        obj_list_ind++;
        continue;
      }

      if(t_ind == 0) {
        tf_w_gt_map[ado_sym] = get<2>(obj_data[obj_list_ind]); // tf_w_ado_gt
        // cout << ado_sym << ": " << tf_w_gt_map[ado_sym] << endl;
        tf_w_est_preslam_map[ado_sym] = get<3>(obj_data[obj_list_ind]); // tf_w_ado_est
        tf_w_ego_gt = Pose3();
      }
      else {
        tf_w_ego_gt = tf_w_gt_map[ado_sym] * tf_ego_ado_gt.inverse(); // gt ego pose in world frame
        tf_w_ego_est = tf_w_est_preslam_map[ado_sym] * tf_ego_ado_est.inverse(); // est ego pose in world frame
      }
      obj_list_ind++;
    }
    tf_w_ego_gt_est_vec.push_back(make_pair(Pose3(tf_w_ego_gt), Pose3(tf_w_ego_est)));
    t_ind++;
  }
}

//////////////////////////////////////////////////////////
// Data Loading Helper Functions
//////////////////////////////////////////////////////////

void load_all_trajectories(map<Symbol, map<double, pair<Pose3, Pose3> > > & all_trajs, set<double> &times, const string path, map<string, obj_param_t> obj_params, double dt_thresh) {
  object_data_vec_t ego_gt_data;
  for(const auto &key_value_pair : obj_params) {
    object_data_vec_t ego_data_gt_single_obj;
    obj_param_t params = key_value_pair.second;
    int obj_id = params.obj_id;
    Symbol sym = Symbol('l', obj_id);

    map<double, pair<Pose3, Pose3> > time_tf_w_ego_map;
    read_gt_datafiles(path + "gt_pose_data_scene_1_" + params.instance_name + ".txt", time_tf_w_ego_map, times);
    all_trajs[sym] = time_tf_w_ego_map;
  }
}

void load_log_files(set<double> &times, object_est_gt_data_vec_t & ado_data, const string path, const string file_base, map<string, obj_param_t> obj_params, double dt_thresh) {
  // for each object, load its est and gt log files to extract pose and time information. combine into a set of all times, and also all the data sorted by time
  for(const auto &key_value_pair : obj_params) {
    object_data_vec_t ado_data_gt, ado_data_est, ego_data_gt;
    object_est_gt_data_vec_t ado_data_single;
    obj_param_t params = key_value_pair.second;
    // string obj_instance_name = params.instance_name; //key_value_pair.first;
    // int obj_id = params.obj_id; //object_id_map[key_value_pair.second];
    cout << "Processing " << params.instance_name << " (id = " << params.obj_id << ")" << endl;

    // read_gt_datafiles(path + "gt_pose_data_scene_1_" + params.instance_name + ".txt", ego_data_gt, times);
    read_data_from_one_log(path + file_base + params.instance_name + "_gt.log", ado_data_gt, times);
    read_data_from_one_log(path + file_base + params.instance_name + "_est.log", ado_data_est, times);
    sync_est_and_gt(ado_data_est, ado_data_gt, ado_data_single, params, dt_thresh);
    ado_data.insert( ado_data.end(), ado_data_single.begin(), ado_data_single.end() ); // combine into 1 vector of all ado data
  }

  // sort by time & object id (the later is just for readability of debugging output, is only tie-break if time are equal)
  std::sort(ado_data.begin(), ado_data.end(), [](const data_tuple& lhs, const data_tuple& rhs) {
    if (get<0>(lhs) == get<0>(rhs)) { // if times are equal
      return get<1>(lhs) < get<1>(rhs); // return true if lhs has lower id
    }
    return get<0>(lhs) < get<0>(rhs); 
  });   // this should sort the vector by time (i.e. first element in each tuple). Tie breaker is object id

}

void read_gt_datafiles(const string fn, map<double, pair<Pose3, Pose3> >& time_tf_w_ego_map, set<double> &times) {
  // space deliminated file: Time (s), ado_name, Ado State tf, Ego State tf. (tfs are x/y/z/r11,r12,r13,...,r33)
  ifstream infile(fn);
  string line, dummy_str;
  double time, x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33;
  Pose3 pose_tf_w_ado_gt, pose_tf_w_ego_gt;
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
    pose_tf_w_ado_gt = Pose3(Rot3(r11, r12, r13, r21, r22, r23, r31, r32, r33), Point3(x, y, z));
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
    pose_tf_w_ego_gt = Pose3(Rot3(r11, r12, r13, r21, r22, r23, r31, r32, r33), Point3(x, y, z));
    // cout << pose_tf_w_ado_gt << endl;
    times.insert(time);
    time_tf_w_ego_map[time] = make_pair(pose_tf_w_ego_gt, pose_tf_w_ado_gt);
    // obj_data.push_back(make_tuple(time, pose_tf_w_ego_gt));
    // obj_data.push_back(make_tuple(time, pose_tf_w_ego_gt.inverse() * pose_tf_w_ado_gt));
    // obj_data.push_back(make_tuple(time, remove_yaw(pose)));
  }
  return;
}

void read_data_from_one_log(const string fn, object_data_vec_t& obj_data, set<double> &times){
  // log file header: Time (s), Ado State GT, Ego State GT, 3D Corner GT (X|Y|Z), Corner 2D Projections GT (r|c), Angled BB (r|c|w|h|ang_deg), Image Segmentation Mode
  // note: the states are position (3), lin vel (3), quat wxyz (4), ang vel (3) (space deliminated)
  ifstream infile(fn);
  string line, s;
  double time, x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz;
  Pose3 pose;
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
    // cout << "qw, qx...qz: " << qw << ", " << qx << ", " << qy << ", " << qz << endl;
    // cout << "quat: " << Quaternion(qw, qx, qy, qz) << endl;
    // cout << "rot: " << Rot3(Quaternion(qw, qx, qy, qz)) << endl;
    // cout << "t: " << Point3(x, y, z) << endl;
    // cout << "pose: " << Pose3(Rot3(Quaternion(qw, qx, qy, qz)), Point3(x, y, z)) << endl;
    pose = Pose3(Rot3(Quaternion(qw, qx, qy, qz)), Point3(x, y, z));
    // NOTE: this conversion from state to pose works the same as that in our python code (verified with test cases)
    times.insert(time);
    obj_data.push_back(make_tuple(time, pose));
    // obj_data.push_back(make_tuple(time, remove_yaw(pose)));
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
          calc_pose_delta(get<1>(data_gt[i]).inverse(), get<1>(data_est[j]), &t_diff, &rot_diff, true);

          if (b_verbose) {
            cout << "\n-------------------------------------------------------------" << endl;
            cout << "a) time = " << t_est << ". id = " << params.obj_id << ".  gt / est diff:  t_delta = " << t_diff << ", r_delta = " << rot_diff << " deg" << endl;
          }
          
          double t_diff2, rot_diff2;
          if (!params.b_rm_roll && !params.b_rm_pitch && !params.b_rm_yaw) {
            calc_pose_delta(get<1>(data_gt[i]).inverse(), get<1>(data_est[j]), &t_diff2, &rot_diff2, true);
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
              Pose3 data_gt_no_yaw = remove_yaw(get<1>(data_gt[i]));
              Pose3 data_est_no_yaw = remove_yaw(get<1>(data_est[i]));
              calc_pose_delta(data_gt_no_yaw.inverse(), data_est_no_yaw, &t_diff2, &rot_diff2, true);
              if (b_verbose) {cout << "c) \t\t\t   w/o yaw:  t_delta2 = " << t_diff2 << ", r_delta2 = " << rot_diff2 << " deg" << endl;}
              calc_pose_delta(data_gt_no_yaw.inverse(), get<1>(data_gt[j]), &t_diff2, &rot_diff2, true);
              if (b_verbose) {cout << "d) \t\t\t   w/ vs. w/o yaw [gt]:   t_diff = " << t_diff2 << ", r_diff = " << rot_diff2 << " deg" << endl;}
              calc_pose_delta(data_est_no_yaw.inverse(), get<1>(data_est[j]), &t_diff2, &rot_diff2, true);
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

void write_results_csv(string fn, map<Symbol, double> ego_time_map, map<Symbol, Pose3> tf_w_gt_map, map<Symbol, Pose3> tf_w_est_preslam_map, map<Symbol, Pose3> tf_w_est_postslam_map, map<Symbol, map<Symbol, pair<Pose3, Pose3> > > tf_ego_ado_maps){
  ofstream myFile(fn);
  double time = 0;
  for(const auto& key_value: tf_w_gt_map) {
    Symbol ego_sym = Symbol(key_value.first);
    Pose3 tf_w_ego_gt = tf_w_gt_map[ego_sym];
    Pose3 tf_w_ego_est_preslam = tf_w_est_preslam_map[ego_sym];
    Pose3 tf_w_ego_est_postslam = tf_w_est_postslam_map[ego_sym];

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
        Symbol ado_sym = key_value.first;
        pair<Pose3, Pose3> pose_gt_est_pair = key_value.second;
        Pose3 tf_ego_ado_gt  = pose_gt_est_pair.first;
        Pose3 tf_ego_ado_est = pose_gt_est_pair.second;
        Pose3 tf_w_ado_gt        = tf_w_ego_gt * tf_ego_ado_gt;
        Pose3 tf_w_ado_est_pre   = tf_w_ego_est_preslam * tf_ego_ado_est;
        Pose3 tf_w_ado_est_post  = tf_w_ego_est_postslam * tf_ego_ado_est;
        myFile << -1 << ", " << ado_sym << ", " << pose_to_string_line(tf_w_ado_gt) << ", " 
                                                << pose_to_string_line(tf_w_ado_est_pre) << ", " 
                                                << pose_to_string_line(tf_w_ado_est_post) << "\n";
      }
    }
  }
  myFile.close();
}

void write_all_traj_csv(string fn, map<Symbol, map<double, pair<Pose3, Pose3> > > & all_trajs) {
  ofstream myFile(fn);
  double time = 0;
  for(const auto& sym_map_key_value: all_trajs) {
    Symbol ado_sym = Symbol(sym_map_key_value.first);
    map<double, pair<Pose3, Pose3> > ado_traj = sym_map_key_value.second;
    for(const auto& time_poses_key_value: ado_traj) {
      double time = time_poses_key_value.first;
      pair<Pose3, Pose3> pose_gt_est_pair = time_poses_key_value.second;
      Pose3 tf_w_ego_gt  = pose_gt_est_pair.first;
      Pose3 tf_w_ego_est = pose_gt_est_pair.second;
      myFile << ado_sym << ", " << time << ", " << pose_to_string_line(tf_w_ego_gt) << ", " << pose_to_string_line(tf_w_ego_est) << pose_to_string_line((tf_w_ego_gt.inverse()) *tf_w_ego_est ) << "\n";
    }
  }
  myFile.close();
}

string pose_to_string_line(Pose3 p){
  string s;
  Point3 t = p.translation();
  Rot3 R = p.rotation();
  Matrix3 M = R.matrix();
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


//////////////////////////////////////////////////////////
// Math Helper Functions
//////////////////////////////////////////////////////////

Pose3 add_init_est_noise(const Pose3 &ego_pose_est) {
  // https://github.com/borglab/gtsam/blob/b1bb0c9ed58f62638c068d7b5332fe7e0e49a29b/examples/SFMExample.cpp
  // noise = np.array([random.uniform(-0.02, 0.02) for i in range(3)]) 
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-0.00005, 0.00005);
  // Pose3 delta(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(dis(gen), dis(gen), dis(gen)));
  Pose3 delta(Rot3::Rodrigues(0.0, 0.0, 0.0), Point3(0.00,0.00,0.00));
  cout << "noise:" << delta << "tf before noise: " << ego_pose_est << endl;
  return ego_pose_est * delta;
  // ego_pose_est = ego_pose_est.compose(delta);
  
  // cout << "tf after noise: " << ego_pose_est << endl;

}

Pose3 add_noise_to_pose3(const Pose3 &pose_in, double dt, double dang) {
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-abs(dt), abs(dt));
  std::uniform_real_distribution<> dis2(-abs(dang), abs(dang));
  Pose3 pose_out = pose_in.compose( Pose3(Rot3::Rodrigues(dis(gen), dis(gen), dis(gen)), 
                                          Point3(dis2(gen), dis2(gen), dis2(gen))) );
  return pose_out;
}

Pose3 remove_yaw(Pose3 P) {
  return Pose3(remove_yaw(P.rotation()), P.translation());
}

Rot3 remove_yaw(Rot3 R) {
  // Goal - recover the XYZ euler angles and set the yaw (rotation about Z) to 0
  // https://eigen.tuxfamily.org/dox/group__Geometry__Module.html#ga17994d2e81b723295f5bc3b1f862ed3b  || https://stackoverflow.com/questions/31589901/euler-to-quaternion-quaternion-to-euler-using-eigen
  /// https://stackoverflow.com/questions/31589901/euler-to-quaternion-quaternion-to-euler-using-eigen
  // https://eigen.tuxfamily.org/dox/group__Geometry__Module.html
  // https://eigen.tuxfamily.org/dox/classEigen_1_1AngleAxis.html
  // roll (X) pitch (Y) yaw (Z) (set Z to 0)


  // Eigen::Vector3f ea = rot3_to_matrix3f(R).eulerAngles(1, 1, 2); 
  // cout << "1) R as ea: " << ea[0] << ", " << ea[1] << ", " << ea[2] << "\n" << endl;
  // Eigen::Quaternionf Q_no_yaw = Eigen::AngleAxisf(ea[0], Eigen::Vector3f::UnitX()) * 
  //                               Eigen::AngleAxisf(ea[1], Eigen::Vector3f::UnitY()) * 
  //                               Eigen::AngleAxisf(0.0,   Eigen::Vector3f::UnitZ());
  // Rot3 R_out = Rot3(Quaternion(Q_no_yaw.w(), Q_no_yaw.x(), Q_no_yaw.y(), Q_no_yaw.z()));
  // Eigen::Vector3f ea_out = rot3_to_matrix3f(R_out).eulerAngles(1, 1, 2); 
  // cout << "2) R_out as ea: " << ea_out[0] << ", " << ea_out[1] << ", " << ea_out[2] << "\n" << endl;
  // return R_out;


  // https://www.mecademic.com/resources/Euler-angles/Euler-angles
  float alpha, beta, gamma;
  Matrix3 M = R.matrix(), M_out;
  double x,y,z,cx,cy,cz,sx,sy,sz;
  sy = M(0,2);

  if ( abs(abs(M(0,2)) - 1) > 0.0001) {
    // not at a singularity
    beta = asin(M(0,2));
    gamma = atan2(-M(0,1), M(0,0));
    alpha = atan2(-M(1,2), M(2,2));
  }
  else {
    alpha = 0;
    beta = M_PI_2;
    gamma = atan2(M(1,0), M(1,1));
  }
  Eigen::Matrix3f rot_matrix = create_rotation_matrix(alpha, beta, 0.0);
  return eigen_matrix3f_to_rot3(rot_matrix);
}

Eigen::Matrix3f rot3_to_matrix3f(Rot3 R) {
  Eigen::Matrix3f m;
  m(0,0) = R.matrix()(0,0); m(0,1) = R.matrix()(0,1); m(0,2) = R.matrix()(0,2);
  m(1,0) = R.matrix()(1,0); m(1,1) = R.matrix()(1,1); m(1,2) = R.matrix()(1,2);
  m(2,0) = R.matrix()(2,0); m(2,1) = R.matrix()(2,1); m(2,2) = R.matrix()(2,2);
  return m;
}

Eigen::Matrix3f create_rotation_matrix(float ax, float ay, float az) {

  // Eigen::Matrix3f R_deltax = np.array([[ 1.             , 0.             , 0.              ],
  //                         [ 0.             , np.cos(Angle_x),-np.sin(Angle_x) ],
  //                         [ 0.             , np.sin(Angle_x), np.cos(Angle_x) ]]);
  // Eigen::Matrix3f R_deltay = np.array([[ np.cos(Angle_y), 0.             , np.sin(Angle_y) ],
  //                         [ 0.             , 1.             , 0               ],
  //                         [-np.sin(Angle_y), 0.             , np.cos(Angle_y) ]]);
  // Eigen::Matrix3f R_deltaz = np.array([[ np.cos(Angle_z),-np.sin(Angle_z), 0.              ],
  //                         [ np.sin(Angle_z), np.cos(Angle_z), 0.              ],
  //                         [ 0.             , 0.             , 1.              ]]);

  Eigen::Matrix3f R_deltax, R_deltay, R_deltaz;
  R_deltax << 1, 0, 0, 0, cos(ax), -sin(ax), 0, sin(ax), cos(ax);
  R_deltay << cos(ay), 0, sin(ay), 0, 1, 0, -sin(ay), 0, cos(ay);
  R_deltaz << cos(az), -sin(az), 0, sin(az), cos(az), 0, 0, 0, 1;
  return R_deltax * R_deltay * R_deltaz;
}

Rot3 eigen_matrix3f_to_rot3(Eigen::Matrix3f M_in) {
  Matrix3 M_out;
  M_out << M_in(0,0), M_in(0,1), M_in(0,2), M_in(1,0), M_in(1,1), M_in(1,2), M_in(2,0), M_in(2,1), M_in(2,2);
  return Rot3(M_out);
}

void calc_pose_delta(const Pose3 & p1, const Pose3 &p2, double *trans_diff, double *rot_diff_rad, bool b_degrees){
  // b_degrees is true if we want degrees, false for radians 
  Pose3 delta = p1.compose(p2);
  *trans_diff = delta.translation().squaredNorm();
  double thresh = 0.001;
  double unit_multiplier = 1;
  double acos_input = (delta.rotation().matrix().trace() - 1) / 2.0;
  if (b_degrees){
    unit_multiplier = 180.0 / M_PI;
  }
  if (acos_input > 1 && (acos_input - 1) < thresh) {
    *rot_diff_rad = 0;
  }
  else if (acos_input > 1) {
    runtime_error("ERROR: cant have acos input > 1!!");
  }
  else if (acos_input < -1 && (abs(acos_input) - 1) < thresh){
    *rot_diff_rad = M_PI * unit_multiplier;
  }
  else if (acos_input < -1 && (abs(acos_input) - 1) < thresh){
    runtime_error("ERROR: cant have acos input < -1!!");
  }
  else {
    *rot_diff_rad = acos( acos_input ) * unit_multiplier;
  }
}



// bool areQuaternionsClose(Quaternion q1, Quaternion q2){
// 	float dot = q1.dot(22); // cos(theta / 2)
// 	if(dot < 0.0f){
// 		return false;					
// 	}
// 	else{
// 		return true;
// 	}
// }

// void average_poses(vector<Pose3>, p_vec){

// }

// void average_quats(Quaternion &cumulative, Quaternion newRotation, Quaternion firstRotation, int addAmount) {
//   //http://wiki.unity3d.com/index.php/Averaging_Quaternions_and_Vectors
//   //Get an average (mean) from more then two quaternions (with two, slerp would be used).
//   //Note: this only works if all the quaternions are relatively close together.
//   //Usage: 
//   //-Cumulative is an ___ which holds all the added x y z and w components.
//   //-newRotation is the next rotation to be added to the average pool
//   //-firstRotation is the first quaternion of the array to be averaged
//   //-addAmount holds the total amount of quaternions which are currently added
//   //This function returns the current average quaternion

// 	float w = 0.0f;
// 	float x = 0.0f;
// 	float y = 0.0f;
// 	float z = 0.0f;
 
// 	//Before we add the new rotation to the average (mean), we have to check whether the quaternion has to be inverted. Because
// 	//q and -q are the same rotation, but cannot be averaged, we have to make sure they are all the same.
// 	if(!areQuaternionsClose(newRotation, firstRotation)){
// 		newRotation = Quaternion(-newRotation.w, -newRotation.x, -newRotation.y, -newRotation.z);	
// 	}
 
// 	//Average the values
// 	float addDet = 1f/(float)addAmount;
// 	cumulative.w += newRotation.w;
// 	w = cumulative.w * addDet;
// 	cumulative.x += newRotation.x;
// 	x = cumulative.x * addDet;
// 	cumulative.y += newRotation.y;
// 	y = cumulative.y * addDet;
// 	cumulative.z += newRotation.z;
// 	z = cumulative.z * addDet;		
// }


