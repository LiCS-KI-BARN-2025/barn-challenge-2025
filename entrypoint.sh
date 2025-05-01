#!/usr/bin/env bash
set -e
source /opt/ros/${ROS_DISTRO}/setup.bash
if [ ! -f "/catkin_ws/devel/setup.bash" ]; then
    echo "Building workspace..."
    catkin build
fi
source /catkin_ws/devel/setup.bash
exec "$@"