#!/usr/bin/env bash
singularity exec -i --nv -n --network=none -p -B `pwd`/src:/catkin_ws/src -B `pwd`/the-barn-challenge:/catkin_ws/the-barn-challenge ${1} /bin/bash /catkin_ws/entrypoint.sh ${@:2}
