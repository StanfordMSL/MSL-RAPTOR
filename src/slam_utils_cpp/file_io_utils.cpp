#include "file_io_utils.h"
#include "Hungarian.h"

using namespace std;
namespace rslam_utils {
                      

  void load_rosbag(vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, pair<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data, int &num_ado_objs,
                    string rosbag_fn, string ego_ns, map<string, obj_param_t> obj_param_map, double dt_thresh, bool b_nocs_data) {
    // OUTPUT:  set<double> times; map<string, object_est_gt_data_vec_t> obj_data  [string is name, object_est_gt_data_vec_t is vector of <time, pose gt, pose est>]
    ROS_INFO("loading rosbag: %s", rosbag_fn.c_str());

    string gt_pose_topic  = "/mavros/vision_pose/pose";
    string est_pose_topic  = "/mavros/local_position/pose";
    string ego_gt_topic_str = "/" + ego_ns + gt_pose_topic;
    string ego_est_topic_str = "/" + ego_ns + est_pose_topic;
    string raptor_topic_str = "/" + ego_ns + "/msl_raptor_state";
    vector<string> ado_instance_names, ado_gt_topic_strs;
    map<string, string> ado_topic_to_name;
    for (const auto & key_val : obj_param_map) {
        string ado_instance_name = key_val.first;
        ado_instance_names.push_back(ado_instance_name);
        ado_gt_topic_strs.push_back("/" + ado_instance_name + gt_pose_topic);
        ado_topic_to_name["/" + ado_instance_name + gt_pose_topic] = ado_instance_name;
    }

    object_data_vec_t ego_data_gt, ego_data_est;   
    map<string, object_data_vec_t> ado_data_gt, ado_data_est;

    rosbag::Bag bag;
    bag.open(rosbag_fn, rosbag::bagmode::Read);
    int num_msg_total = 0;
    double time = 0.0, time0 = -1, ave_dt = 0, last_time = 0, last_time_debug = -1;
    geometry_msgs::PoseStamped::ConstPtr geo_msg = nullptr;
    map<string, geometry_msgs::Pose> prev_gt_pose;
    double prev_gt_time = -100000000;
    map<string, vector<pair<string, geometry_msgs::Pose>>> instances_and_poses_by_class_map; 

    ///////
    // loop through all the gt data to find how many of each class type we see (and also what their FIRST pose is)
    cout << "WARNING!!! Using the gt data to get perfect gt's for corespondence. In realtime, will need to build structure of seen objects and poses iteratively!" << endl;
    set<string> seen_instances;
    for(rosbag::MessageInstance const m: rosbag::View(bag)) { // DUMMY FOR LOOP TO INITIALIZE THE GT POSES FOR HUNGARIAN ALGO
      if (m.getTopic() == ego_gt_topic_str || ("/" + m.getTopic() == ego_gt_topic_str)) {
          continue;
      }
      else if (m.getTopic() == ego_est_topic_str || ("/" + m.getTopic() == ego_est_topic_str)) {
          continue;
      }
      else if (m.getTopic() == raptor_topic_str || ("/" + m.getTopic() == raptor_topic_str)) {
          continue;
      }
      else {
        for (const auto &  topic_str : ado_gt_topic_strs) {
          if (m.getTopic() == topic_str || ("/" + m.getTopic() == topic_str)) {
            geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
            if (geo_msg != nullptr) {
              prev_gt_pose[ado_topic_to_name[topic_str]] = geo_msg->pose;
              string instance_str = ado_topic_to_name[topic_str];
              if( !seen_instances.count(instance_str) ) { // we have not seen this specific object before
                string cls_str = obj_param_map[instance_str].class_name;
                // if( instances_and_poses_by_class_map.find(cls_str) == instances_and_poses_by_class_map.end() ) {
                if( !instances_and_poses_by_class_map.count(cls_str) ) { // first time seeing this class - make new vector
                  // map<string, geometry_msgs::Pose> tmp_map;
                  // tmp_map[obj_param_map[ado_topic_to_name[topic_str]].instance_name] = geo_msg->pose;
                  vector<pair<string, geometry_msgs::Pose>> tmp_vec;
                  tmp_vec.emplace_back(instance_str, geo_msg->pose);
                  instances_and_poses_by_class_map[cls_str] = tmp_vec; // first time, need to initialize vector
                }
                else { 
                  instances_and_poses_by_class_map[cls_str].emplace_back(instance_str, geo_msg->pose); // save last pose
                }
                seen_instances.insert(instance_str);
              }
              break;
            }
          }
        }
      }
    }
    for (const auto & key_val : prev_gt_pose) {
      cout << "obj instance: " << key_val.first << endl;
    }

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
                ego_data_gt.emplace_back(time, ros_geo_pose_to_gtsam(geo_msg->pose));
            }
        }
        else if (m.getTopic() == ego_est_topic_str || ("/" + m.getTopic() == ego_est_topic_str)) {
            geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
            if (geo_msg != nullptr) {
                ego_data_est.emplace_back(time, ros_geo_pose_to_gtsam(geo_msg->pose));
            }
        }
        else if (m.getTopic() == raptor_topic_str || ("/" + m.getTopic() == raptor_topic_str)) {
          // THIS IS MSL-RAPTOR data for estimates of the tracked object poses
          //     Note: we dont know the object's instance name, just the class at this point
          msl_raptor::TrackedObjects::ConstPtr raptor_msg = m.instantiate<msl_raptor::TrackedObjects>();
          
          if (raptor_msg != nullptr && (raptor_msg->tracked_objects).size() > 0) {

            // Seperate observed poses by class
            map<string, vector<geometry_msgs::Pose>> observed_poses_by_class;
            for (const auto & tracked_obj : raptor_msg->tracked_objects) {
              geometry_msgs::Pose tracked_pose = tracked_obj.pose.pose;
              if( observed_poses_by_class.count(tracked_obj.class_str) == 0 ) { 
                // first observance of this object, need to initialize vector
                vector<geometry_msgs::Pose> pose_vec{tracked_pose};
                observed_poses_by_class[tracked_obj.class_str] = pose_vec; 
              }
              else {
                observed_poses_by_class[tracked_obj.class_str].push_back(tracked_pose); // save last pose
              }
            }
            for (const auto & class_pose_vec : observed_poses_by_class) {
              string cls = class_pose_vec.first;
              vector<geometry_msgs::Pose> est_pose_vec = class_pose_vec.second;
              vector<pair<string, geometry_msgs::Pose>> instance_gt_pose_vec = instances_and_poses_by_class_map[cls];

              if (instance_gt_pose_vec.size() == 1) {
                // No need for hungarian algo, just 1 object of this class in our environment!
                string instance_name = instance_gt_pose_vec[0].first;
                // geometry_msgs::Pose pose = instance_gt_pose_vec[0].second; // THIS IS WRONG!!!
                geometry_msgs::Pose pose = est_pose_vec[0]; 
                assert(est_pose_vec.size()==1);
                ado_data_est[instance_name].emplace_back(time, ros_geo_pose_to_gtsam(pose));
              }
              else { // use Hungarian algo!
              // For each class seen, if more than 1 object use hungarian algo with the gt data
              //    hungarian algo: 1) build cost matrix 2) solve assignment
                vector< vector<double> > costMatrix; // rows: # object instances seen this timestep, col: total # of possible instances of this class
                bool b_print_out_hungarian_algo_details = false;
                int row_index = 0; // index of currently seen object instances i.e. est_pose_vec

                vector<string> candidate_ado_names;
                for (const auto & tracked_pose : est_pose_vec) {
                  costMatrix.push_back({});
                  for(const auto & instance_str_pose_pair : instance_gt_pose_vec) {
                    if(row_index == 0) {
                      candidate_ado_names.push_back(instance_str_pose_pair.first); // only fill this once
                    }
                    geometry_msgs::Pose gt_pose = instance_str_pose_pair.second;
                    double dist = sqrt( (gt_pose.position.x - tracked_pose.position.x) * (gt_pose.position.x - tracked_pose.position.x) +
                                        (gt_pose.position.y - tracked_pose.position.y) * (gt_pose.position.y - tracked_pose.position.y) +
                                        (gt_pose.position.z - tracked_pose.position.z) * (gt_pose.position.z - tracked_pose.position.z) );
                    costMatrix[row_index].push_back(dist);
                  }
                  row_index++;
                }
                HungarianAlgorithm HungAlgo;
                vector<int> assignment;  // values here range from 0 to num_cols - 1 (the length of this variable will = num rows)
                double cost = HungAlgo.Solve(costMatrix, assignment);

                // Show full assignemnts cost matrix:
                if(b_print_out_hungarian_algo_details) {
                  cout << "assignements:  ";
                  for (int i = 0; i < assignment.size(); i++) {
                    cout << assignment[i] << ", ";
                  }
                  cout << endl;
                  for (unsigned int x = 0; x < costMatrix.size(); x++) {
                    // cout << "\nrow: " << x << ", candidate_ado_names[row] = " << endl;
                    cout << "class pose est " << x << ": ";
                    for(int c = 0; c < costMatrix[x].size(); c++) {
                      cout << costMatrix[x][c] << " (" << candidate_ado_names[c] << ")";
                      if ( c < costMatrix[x].size() - 1) {
                        cout << ", ";
                      }
                    }
                    cout << endl;
                  }
                }
                for (unsigned int x = 0; x < costMatrix.size(); x++) {
                  if(assignment[x] >= 0) { // costMatrix.size(); = # of rows = length(est_pose_vec)
                    if(b_print_out_hungarian_algo_details) {// show info for chosen value:
                      cout << "class pose est " << x << " assigned to: " << candidate_ado_names[assignment[x]] << " (column # " << assignment[x] << ") for row cost = " << costMatrix[x][assignment[x]] << " ||| ";
                    }
                    ado_data_est[candidate_ado_names[assignment[x]]].emplace_back(time, ros_geo_pose_to_gtsam(est_pose_vec[x]));
                  }
                  else {
                    if(b_print_out_hungarian_algo_details) {
                      cout << "class pose est " << x << " NOT ASSIGNED (designated as noise)" << " ||| ";
                    }
                  }
                }
                if(b_print_out_hungarian_algo_details) {
                  std::cout << "\n total cost: " << cost << std::endl;
                  cout << "" << endl;
                }
              }
            }            
          }
        }
        else {
          for (const auto &  topic_str : ado_gt_topic_strs) {
            if (m.getTopic() == topic_str || ("/" + m.getTopic() == topic_str)) {
              geo_msg = m.instantiate<geometry_msgs::PoseStamped>();
              if (geo_msg != nullptr) {
                // if( ado_topic_to_name[topic_str] == "bowl_green_msl") {
                //   cout << "new gt message for " << ado_topic_to_name[topic_str] << " at " << time*1000 << endl;
                //   if( abs(last_time_debug - time) < 0.00001) {
                //     cout << endl;
                //   }
                //   last_time_debug = time;
                // }
                ado_data_gt[ado_topic_to_name[topic_str]].emplace_back(time, remove_yaw(ros_geo_pose_to_gtsam(geo_msg->pose)));
                prev_gt_pose[ado_topic_to_name[topic_str]] = geo_msg->pose;
                break;
              }
            }
          }
      }
      num_msg_total++;
    }
    num_ado_objs = ado_data_gt.size();
    cout << "num messages = " << num_msg_total << ", # ado objects: " << num_ado_objs << endl;
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

    vector<pair<double, map<string, pair<gtsam::Pose3, gtsam::Pose3>>>> ado_data_grouped;
    group_ado_measurements_by_time(ado_data_grouped, ado_data, dt_thresh);

    zip_data_by_ego(raptor_data, ego_data, ado_data_grouped, dt_thresh);

    // // trim_data_range(raptor_data, 20, 40);
    write_batch_slam_inputs_csv("/mounted_folder/test_graphs_gtsam/batch_input1.csv", raptor_data, obj_param_map);
    // if (1 || b_nocs_data) {
    //   fn = "/mounted_folder/test_graphs_gtsam/all_trajs.csv";
    //   map<gtsam::Symbol, map<double, pair<gtsam::Pose3, gtsam::Pose3> > > all_trajs;
    //   write_all_traj_csv(fn, raptor_data, all_trajs, obj_param_map, num_ado_objs);
    //   // convert_data_to_static_obstacles(raptor_data, num_ado_objs);

    //   // fn = "/mounted_folder/test_graphs_gtsam/batch_input2.csv";
    //   // write_batch_slam_inputs_csv(fn, raptor_data, obj_param_map);
    // }
    return;
  }

  void trim_data_range(vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, pair<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data, int start_ind, int end_ind) {
    vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, pair<gtsam::Pose3, gtsam::Pose3> > > > raptor_data_new;
    for (int i = start_ind; i < end_ind+1; i++) {
      raptor_data_new.push_back(raptor_data[i]);
    }
    raptor_data.clear();
    raptor_data = raptor_data_new;
  }

  void group_ado_measurements_by_time(vector<pair<double, map<string, pair<gtsam::Pose3, gtsam::Pose3>>>> &ado_data_grouped,
                                        const map<string, object_est_gt_data_vec_t> &ado_data, double dt_thresh) {
    // first find the lowest time index
    map<string, int> ado_idxs;  // ado name, ado index, 
    double current_time = 1000000000;
    for (const auto & key_val : ado_data) {
      ado_idxs[key_val.first] = 0;
      double time = get<0>(key_val.second[0]);
      if (time < current_time) {
        current_time = time;
      }
    } // done finding lowest time index

    // Next, step through the times grouping them if they fall within dt_thresh. Stop when all are grouped
    bool b_finished = false;
    while(!b_finished) {
      map<string, pair<gtsam::Pose3, gtsam::Pose3>> measurement;
      double meas_time = 0, next_time = 1000000000;
      b_finished = true; // this is last round unless one of our ado vectors still has more left
      for (const auto & key_val : ado_data) {
        string ado_name = key_val.first;
        if (ado_idxs[ado_name] < 0) {
          continue; // this means we reached the end of this vector
        }
        
        object_est_gt_data_vec_t ado_data_single = key_val.second;
        data_tuple ado_data_tuple = ado_data_single[ado_idxs[ado_name]];
        double this_ado_time = get<0>(ado_data_tuple);
        if ( dt_thresh > abs(this_ado_time - current_time) ) {
          // this ado data is part of this msl_raptor iteration
          measurement[ado_name] = make_pair(get<2>(ado_data_tuple), get<3>(ado_data_tuple));
          meas_time += this_ado_time; // summed here because will be averaged at end of loop
          if (ado_idxs[ado_name] + 1 < ado_data_single.size()) {
            ado_idxs[ado_name]++;
            b_finished = false; // do at least 1 more round
          }
          else {
            ado_idxs[ado_name] = -1; // mark this as finished
          }
        }
        if ( (ado_idxs[ado_name] >= 0) && (get<0>(ado_data_single[ado_idxs[ado_name]]) < next_time) ) {
          if ( abs(get<0>(ado_data_single[ado_idxs[ado_name]]) - get<0>(ado_data_single[ado_idxs[ado_name] - 1])) < 0.00001) {
            cout << "WE HAVE TWO RAPTOR MEAUSMRENTS WITH THE SAME TIME!! THIS SHOULDNT HAPPEN!" << endl;
            assert(false);
          }
          next_time = get<0>(ado_data_single[ado_idxs[ado_name]]); // get the next closest ado time to use for next raptor iteration
          if(abs(next_time - current_time) < 0.001) {
            cout << ado_idxs[ado_name] << endl; // DEBUG - this shouldnt happen
          }
        }
      }
      ado_data_grouped.emplace_back( meas_time / ((double)measurement.size()), measurement );
      current_time = next_time;
    } // done grouping measurements
  }

  
  void zip_data_by_ego(vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, pair<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data, 
                        object_est_gt_data_vec_t ego_data, const vector<pair<double, map<string, pair<gtsam::Pose3, gtsam::Pose3>>>> &ado_data_grouped,
                        double dt_thresh) {
    // Combine all data into a single data structure that can be looped over. Each element simulates a potential "measurement" from msl_raptor

    int ego_idx = 0;
    for (const auto & key_val : ado_data_grouped) {
      double ado_time = key_val.first;
      map<string, pair<gtsam::Pose3, gtsam::Pose3>> measurements = key_val.second;
      
      // get_interpolated_ego_pose(ado_time, ego_data, ego_idx);
      double last_ego_time  = get<0>(ego_data[ego_idx]);
      gtsam::Pose3 tf_ego_gt = get<2>(ego_data[ego_idx]);
      gtsam::Pose3 tf_ego_est = get<2>(ego_data[ego_idx]);
      gtsam::Pose3 prev_gt  = get<2>(ego_data[ego_idx]);
      gtsam::Pose3 prev_est = get<3>(ego_data[ego_idx]);

      for(int i = ego_idx; i < ego_data.size(); i++) {
        data_tuple ego_tuple = ego_data[i];
        double ego_time = get<0>(ego_tuple);
        if (last_ego_time <= ado_time && ado_time <= ego_time ) {
          // found surounding ego times
          double s = (ado_time - last_ego_time) / (ego_time - last_ego_time);
          tf_ego_gt  = interp_pose(prev_gt,  get<2>(ego_data[i]), s);
          tf_ego_est = interp_pose(prev_est, get<2>(ego_data[i]), s);
          break; // return
        }
        // else...
        last_ego_time  = ego_time;
        prev_gt  = get<2>(ego_data[i]);
        prev_est = get<3>(ego_data[i]);
      }
      raptor_data.emplace_back(ado_time, tf_ego_gt, tf_ego_est, measurements);
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

  void sync_est_and_gt(object_data_vec_t data_est, object_data_vec_t data_gt, object_est_gt_data_vec_t& data, obj_param_t params, double dt_thresh) {
    // for each ado datapoint (after gt data has begun), interpolate to gt poses to find where the ado was taken

    bool b_verbose = true;
    double t_gt, t_est, t_prev_gt = 0, t_prev_est = 0;
    uint next_gt_time_ind = 0;
    gtsam::Pose3 prev_gt, prev_est;
    int gt_ind = 0;
    double t_gt0 = get<0>(data_gt[gt_ind]), prev_est_t = -1434.324234;

    for (uint i = 0; i < data_est.size()-1; i++) {
      t_est = get<0>(data_est[i]);
      if (t_est < t_gt0) {
        continue;
      }

      double t_gt1, t_gt2, s;
      while (get<0>(data_gt[gt_ind]) < t_est) {
        t_gt1 = get<0>(data_gt[gt_ind]);
        prev_gt = get<1>(data_gt[gt_ind]);
        gt_ind++;
        if(gt_ind >= data_gt.size()) {
          break;
        }
      }
      t_gt2 = get<0>(data_gt[gt_ind]);
      s = (t_est - t_gt1) / (t_gt2 - t_gt1);
      gtsam::Pose3 tf_gt = rslam_utils::interp_pose(prev_gt, get<1>(data_gt[gt_ind]), s);
      data.emplace_back(t_est, params.obj_id, tf_gt, get<1>(data_est[i]));
      prev_est_t = t_est;
    }
  }

  void convert_data_to_static_obstacles(vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, pair<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data, int num_ado_objs) {
    // assume all ado objects are static and calculate the world pose of ego
    vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, pair<gtsam::Pose3, gtsam::Pose3> > > > data_out;
    
    map<string, gtsam::Pose3> tf_w_ado0_gt, tf_w_ado0_est;
    get_tf_w_ado_for_all_objects(raptor_data, tf_w_ado0_gt, tf_w_ado0_est);
    // int num_ado_obj_seen = get_tf_w_ado_for_all_objects(raptor_data, tf_w_ado0_gt, tf_w_ado0_est, num_ado_objs);
    // assert(num_ado_obj_seen == num_ado_objs);
    // for (const auto & key_val : tf_w_ado0_gt) {
    //   string ado_name = key_val.first;
    //   gtsam::Pose3 tf_w_ado_gt = key_val.second;
    //   gtsam::Pose3 tf_w_ado_est = tf_w_ado0_est[ado_name];
    //   cout << ado_name << "... gt: " << tf_w_ado_gt << endl;
    //   // cout << ado_name << "... gt: " << tf_w_ado_gt << " est: " << tf_w_ado_est << endl;
    // }
    
    for (const auto & rstep : raptor_data) {
      double t = get<0>(rstep);
      gtsam::Pose3 tf_ego_w_gt_unrect = get<1>(rstep).inverse();
      gtsam::Pose3 tf_ego_w_est_unrect = get<2>(rstep).inverse();
      map<string, pair<gtsam::Pose3, gtsam::Pose3> > measurements_unrect = get<3>(rstep);
      map<string, pair<gtsam::Pose3, gtsam::Pose3> > measurements_out;

      gtsam::Pose3 tf_w_ego_gt, tf_w_ego_est;
      for (const auto & key_val : measurements_unrect) {
        string ado_name = key_val.first;
        pair<gtsam::Pose3, gtsam::Pose3> ado_w_gt_est_pair_unrect = key_val.second;
        gtsam::Pose3 tf_ego_ado_gt  = tf_ego_w_gt_unrect  * ado_w_gt_est_pair_unrect.first;  // (tf_w_ego_gt.inverse()) * tf_w_ado_gt;
        gtsam::Pose3 tf_ego_ado_est = tf_ego_w_est_unrect * ado_w_gt_est_pair_unrect.second; // (tf_w_ego_est.inverse()) * tf_w_ado_est;

        if(measurements_out.empty()) { 
          tf_w_ego_gt  = tf_w_ado0_gt[ado_name]  * (tf_ego_ado_gt.inverse());
          tf_w_ego_est = tf_w_ado0_est[ado_name] * (tf_ego_ado_est.inverse());
        }
        gtsam::Pose3 tf_w_ado_gt  = tf_w_ego_gt  * tf_ego_ado_gt;
        gtsam::Pose3 tf_w_ado_est = tf_w_ego_est * tf_ego_ado_est;
        measurements_out[ado_name] = make_pair(tf_w_ado_gt, tf_w_ado_est);
      }
      data_out.emplace_back(t, tf_w_ego_gt, tf_w_ego_est, measurements_out);
    }
    raptor_data = data_out;
  }

  void get_tf_w_ado_for_all_objects(const vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, pair<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data, 
                                    map<string, gtsam::Pose3> &tf_w_ado0_gt, map<string, gtsam::Pose3> &tf_w_ado0_est) {

    // fill in tf_w_ado0_est and tf_w_ado0_gt. This can be tricky when we only have relative measurements and we dont see an object on the first timestep
    
    // first loop over data to fill in these maps
    for (const auto & rstep : raptor_data) {
      map<string, pair<gtsam::Pose3, gtsam::Pose3> > measurements = get<3>(rstep);
      for (const auto & key_val : measurements) { // name, pair<Pose3, Pose3>
        string ado_name = key_val.first;
        pair<gtsam::Pose3, gtsam::Pose3> ado_w_gt_est_pair = key_val.second;
        gtsam::Pose3 tf_w_ado_gt = ado_w_gt_est_pair.first;
        gtsam::Pose3 tf_w_ado_est = ado_w_gt_est_pair.second;
        if(tf_w_ado0_gt.count(ado_name) == 0) {
          // this means we have not seen this ado object before
          tf_w_ado0_gt[ado_name] = tf_w_ado_gt;
          tf_w_ado0_est[ado_name] = tf_w_ado_est;
        } // end if we have seen this ado object before
      } // end loop over ado objects at this time step
    } // end loop over all data
  }


  void write_batch_slam_inputs_csv(string fn, vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, pair<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data, 
                                    map<string, obj_param_t> obj_param_map) {
    ofstream myFile(fn);
    int t_ind = 0;
    for (const auto & raptor_step : raptor_data ) {
      double time = get<0>(raptor_step);
      gtsam::Pose3 tf_w_ego_gt  = get<1>(raptor_step);
      gtsam::Pose3 tf_w_ego_est = get<2>(raptor_step);

      int ego_pose_index = 1 + t_ind;
      gtsam::Symbol ego_sym = gtsam::Symbol('x', ego_pose_index);

      myFile << time << ", " << ego_sym << ", " << pose_to_string_line(tf_w_ego_gt) << ", " 
                                                << pose_to_string_line(tf_w_ego_est) << ", " 
                                                << pose_to_string_line(tf_w_ego_est) << "\n";
      map<string, pair<gtsam::Pose3, gtsam::Pose3> > measurements = get<3>(raptor_step);
      for (const auto & key_val : measurements) {
        string ado_name = key_val.first;
        gtsam::Symbol ado_sym = gtsam::Symbol('l', obj_param_map[ado_name].obj_id);
        gtsam::Pose3 tf_w_ado_gt = key_val.second.first;
        gtsam::Pose3 tf_w_ado_est = key_val.second.second;

        myFile << -1 << ", " << ado_sym << ", " << pose_to_string_line(tf_w_ado_gt) << ", " 
                                                << pose_to_string_line(tf_w_ado_est) << ", " 
                                                << pose_to_string_line(tf_w_ado_est) << "\n";
      }
      t_ind++;  
    }
    myFile.close();
  }

  void write_results_csv(string fn, const vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, pair<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data, 
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
      gtsam::Pose3 tf_w_ego_est_pre  = tf_w_est_preslam_map[ego_sym];
      gtsam::Pose3 tf_w_ego_est_post = tf_w_est_postslam_map[ego_sym];

      myFile << time << ", " << ego_sym << ", " << pose_to_string_line(tf_w_ego_gt) << ", " 
                                                << pose_to_string_line(tf_w_ego_est_pre) << ", " 
                                                << pose_to_string_line(tf_w_ego_est_post) << "\n";

      map<gtsam::Symbol, pair<gtsam::Pose3, gtsam::Pose3> > ado_w_gt_est_pair_map = tf_ego_ado_maps[ego_sym];

      for (const auto & single_ado_meas : ado_w_gt_est_pair_map) {
        gtsam::Symbol ado_sym = single_ado_meas.first;
        gtsam::Pose3 tf_ego_ado_gt  = single_ado_meas.second.first;
        gtsam::Pose3 tf_ego_ado_est = single_ado_meas.second.second;

        gtsam::Pose3 tf_w_ado_gt        = tf_w_ego_gt * tf_ego_ado_gt;
        gtsam::Pose3 tf_w_ado_est_pre   = tf_w_ego_est_pre * tf_ego_ado_est;
        gtsam::Pose3 tf_w_ado_est_post  = tf_w_ego_est_post * tf_ego_ado_est;


        myFile << -1 << ", " << ado_sym << ", " << pose_to_string_line(tf_w_ado_gt) << ", " 
                                                << pose_to_string_line(tf_w_ado_est_pre) << ", " 
                                                << pose_to_string_line(tf_w_ado_est_post) << "\n";
        // myFile << -1 << ", " << ado_sym << ", " << pose_to_string_line(tf_w_ado_gt) << ", " 
        //                                         << pose_to_string_line(tf_w_est_preslam_map[ado_sym]) << ", " 
        //                                         << pose_to_string_line(tf_w_est_postslam_map[ado_sym]) << "\n";
      }
      t_ind++;
    }
    myFile.close();
  }

  void write_all_traj_csv(string fn, const vector<tuple<double, gtsam::Pose3, gtsam::Pose3, map<string, pair<gtsam::Pose3, gtsam::Pose3> > > > &raptor_data,
                            map<gtsam::Symbol, map<double, pair<gtsam::Pose3, gtsam::Pose3> > > & all_trajs,
                            map<string, obj_param_t> &obj_param_map, int num_ado_objs) {
    // generate all trajectories
    map<string, gtsam::Pose3> tf_w_ado0_gt, tf_w_ado0_est;
    get_tf_w_ado_for_all_objects(raptor_data, tf_w_ado0_gt, tf_w_ado0_est);
    
    ofstream myFile(fn);

    for (const auto & key_val : tf_w_ado0_gt ) {
      string ado_name = key_val.first;
      gtsam::Symbol ado_sym = gtsam::Symbol('l', obj_param_map[ado_name].obj_id);

      map<double, pair<gtsam::Pose3, gtsam::Pose3> > one_traj;
      int t_ind = 0;
      for (const auto & rstep : raptor_data ) {
        map<string, pair<gtsam::Pose3, gtsam::Pose3> > measurements = get<3>(rstep);
        if(measurements.find(ado_name) == measurements.end()) {
          continue; // this object wasnt seen this iteration
        }
        double time = get<0>(rstep);
        gtsam::Pose3 tf_w_ado_gt_unrect  = measurements[ado_name].first;
        gtsam::Pose3 tf_w_ado_est_unrect = measurements[ado_name].second;
        gtsam::Pose3 tf_w_ego_gt_unrect = get<1>(rstep);
        gtsam::Pose3 tf_w_ego_est_unrect = get<2>(rstep);
        gtsam::Pose3 tf_ego_ado_gt = tf_w_ego_gt_unrect.inverse() * tf_w_ado_gt_unrect;
        gtsam::Pose3 tf_ego_ado_est = tf_w_ego_est_unrect.inverse() * tf_w_ado_est_unrect;

        gtsam::Pose3 tf_w_ego_gt  = tf_w_ado0_gt[ado_name]  * tf_ego_ado_gt.inverse();
        gtsam::Pose3 tf_w_ego_est = tf_w_ado0_est[ado_name] * tf_ego_ado_est.inverse();
        myFile << ado_sym << ", " << time << ", " << pose_to_string_line(tf_w_ego_gt) << ", " 
                                                  << pose_to_string_line(tf_w_ego_est) << ", " 
                                                  << pose_to_string_line(tf_w_ego_gt * tf_ego_ado_est ) << "\n";
        one_traj[time] = make_pair(tf_w_ego_gt, tf_w_ego_est);
      }
      all_trajs[ado_sym] = one_traj;
    }
    myFile.close();
    
    
    
    // ofstream myFile(fn);
    // double time = 0;
    // for(const auto& sym_map_key_value: all_trajs) {
    //   gtsam::Symbol ado_sym = gtsam::Symbol(sym_map_key_value.first);
    //   map<double, pair<gtsam::Pose3, gtsam::Pose3> > ado_traj = sym_map_key_value.second;
    //   for(const auto& time_poses_key_value: ado_traj) {
    //     double time = time_poses_key_value.first;
    //     pair<gtsam::Pose3, gtsam::Pose3> pose_gt_est_pair = time_poses_key_value.second;
    //     gtsam::Pose3 tf_w_ego_gt  = pose_gt_est_pair.first;
    //     gtsam::Pose3 tf_w_ego_est = pose_gt_est_pair.second;
    //     myFile << ado_sym << ", " << time << ", " << pose_to_string_line(tf_w_ego_gt) << ", " << pose_to_string_line(tf_w_ego_est) << pose_to_string_line((tf_w_ego_gt.inverse()) *tf_w_ego_est ) << "\n";
    //   }
    // }
    // myFile.close();
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

}  // end of namespace rslam_utils