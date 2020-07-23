#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Int32.h>
#include <msl_raptor/AngledBbox.h>
#include <msl_raptor/AngledBbox.h>
#include <msl_raptor/TrackedObject.h>
#include <msl_raptor/TrackedObjects.h>

namespace rslam_utils {
    void load_raptor_output_rosbag(std::string rosbag_fn);
    void load_gt_rosbag(std::string);
}