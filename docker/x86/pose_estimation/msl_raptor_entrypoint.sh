#!/bin/bash
set -e

# start avahi daemon
/etc/init.d/dbus start &>/dev/null
service avahi-daemon start &>/dev/null

# setup ros environment
source "/root/msl_raptor_ws/devel/setup.bash"
export PYTHONPATH="/root/msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask:$PYTHONPATH:/root/msl_raptor_ws/src/msl_raptor/src/front_end/yolov3:/root/msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask/experiments/siammask_sharp:/root/python3_ws/install/lib/python3/dist-packages"
source "/usr/local/bin/nvidia_entrypoint.sh"

exec "$@"
