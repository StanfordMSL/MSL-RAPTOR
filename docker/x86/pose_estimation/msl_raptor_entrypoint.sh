#!/bin/bash
set -e

# setup ros environment
source "/root/msl_raptor_ws/devel/setup.bash"
export PYTHONPATH="/root/msl_raptor_ws/src/msl_raptor/src/front-end/SiamMask:$PYTHONPATH:/root/msl_raptor_ws/src/msl_raptor/src/front-end/yolov3:/root/msl_raptor_ws/src/msl_raptor/src/front-end/SiamMask/experiments/siammask_sharp:/root/python3_ws/install/lib/python3/dist-packages"
source "/usr/local/bin/nvidia_entrypoint.sh"
exec "$@"