#!/bin/bash
set -e

# setup ros environment
source "/root/msl_raptor_ws/devel/setup.bash"
exec "$@"
