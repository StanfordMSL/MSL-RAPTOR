#!/bin/bash
set -e

# setup ros environment
source "/root/python3_ws/install/setup.bash"

exec "$@"