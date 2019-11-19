#!/bin/bash
set -e

# setup ros environment
source "/root/cv_bridge_py3_ws/install/setup.bash"
exec "$@"