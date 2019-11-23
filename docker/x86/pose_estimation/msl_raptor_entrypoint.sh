#!/bin/bash
set -e

# setup ros environment
source "/root/msl_raptor_ws/devel/setup.bash"
export PYTHONPATH="$PYTHONPATH:/root/python3_ws/install/lib/python3/dist-packages"
exec "$@"
