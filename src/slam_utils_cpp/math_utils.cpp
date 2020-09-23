#include "math_utils.h"

//////////////////////////////////////////////////////////
// Math Helper Functions
//////////////////////////////////////////////////////////

namespace rslam_utils {

gtsam::Pose3 interp_pose(gtsam::Pose3 tf1, gtsam::Pose3 tf2, double s) {
  // tf1 is be the earlier pose, tf2 the later. s is between 0 and 1
  if (s < 0 || s > 1) {
    std::runtime_error("interpolation value must be between 0 and 1");
  }
  gtsam::Pose3 tf_out;
  gtsam::Rot3 Rout = tf1.rotation().slerp(s, tf2.rotation());
  gtsam::Point3 pout = s*(tf2.translation() - tf1.translation()) + tf1.translation();
  return gtsam::Pose3(Rout, pout);
}

gtsam::Pose3 add_init_est_noise(const gtsam::Pose3 &ego_pose_est, double dt) {
  // https://github.com/borglab/gtsam/blob/b1bb0c9ed58f62638c068d7b5332fe7e0e49a29b/examples/SFMExample.cpp
  // noise = np.array([random.uniform(-0.02, 0.02) for i in range(3)]) 
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-abs(dt), abs(dt));
  gtsam::Pose3 delta(gtsam::Rot3::Rodrigues(0.0, 0.0, 0.0), gtsam::Point3(dis(gen), dis(gen), dis(gen)));
  // gtsam::Pose3 delta(gtsam::Rot3::Rodrigues(0.0, 0.0, 0.0), gtsam::Point3(0.00,0.00,0.00));
  // cout << "noise:" << delta << "tf before noise: " << ego_pose_est << endl;
  return ego_pose_est * delta;
  // ego_pose_est = ego_pose_est.compose(delta);
  
  // cout << "tf after noise: " << ego_pose_est << endl;

}

gtsam::Pose3 add_noise_to_pose3(const gtsam::Pose3 &pose_in, double dt, double dang) {
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-abs(dt), abs(dt));
  std::uniform_real_distribution<> dis2(-abs(dang), abs(dang));
  gtsam::Pose3 pose_out = pose_in.compose( gtsam::Pose3(gtsam::Rot3::Rodrigues(dis(gen), dis(gen), dis(gen)), 
                                           gtsam::Point3(dis2(gen), dis2(gen), dis2(gen))) );
  return pose_out;
}

gtsam::Pose3 remove_yaw(gtsam::Pose3 P) {
  return gtsam::Pose3(remove_yaw(P.rotation()), P.translation());
}

gtsam::Rot3 remove_yaw(gtsam::Rot3 R) {
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
  gtsam::Matrix3 M = R.matrix(), M_out;
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

Eigen::Matrix3f rot3_to_matrix3f(gtsam::Rot3 R) {
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

gtsam::Rot3 eigen_matrix3f_to_rot3(Eigen::Matrix3f M_in) {
  gtsam::Matrix3 M_out;
  M_out << M_in(0,0), M_in(0,1), M_in(0,2), M_in(1,0), M_in(1,1), M_in(1,2), M_in(2,0), M_in(2,1), M_in(2,2);
  return gtsam::Rot3(M_out);
}

void calc_pose_delta(const gtsam::Pose3 & p1, const gtsam::Pose3 &p2, double *trans_diff, double *rot_diff_rad, bool b_degrees){
  // b_degrees is true if we want degrees, false for radians 
  gtsam::Pose3 delta = p1.compose(p2);
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

}