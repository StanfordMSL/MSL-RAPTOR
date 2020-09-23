#pragma once

// GTSAM Includes
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>

#include "shared_imports.h"

namespace rslam_utils {
    gtsam::Pose3 interp_pose(gtsam::Pose3 tf1, gtsam::Pose3 tf2, double s);
    gtsam::Pose3 add_init_est_noise(const gtsam::Pose3 &ego_pose_est);
    gtsam::Pose3 add_noise_to_pose3(const gtsam::Pose3 &pose_in, double dt, double dang);
    gtsam::Pose3 remove_yaw(gtsam::Pose3 P);
    gtsam::Rot3 remove_yaw(gtsam::Rot3 R);
    Eigen::Matrix3f Rot3_to_matrix3f(gtsam::Rot3 R);
    Eigen::Matrix3f create_rotation_matrix(float ax, float ay, float az);
    gtsam::Rot3 eigen_matrix3f_to_rot3(Eigen::Matrix3f M_in);
    void calc_pose_delta(const gtsam::Pose3 & p1, const gtsam::Pose3 &p2, double *trans_diff, double *rot_diff_rad, bool b_degrees);
}