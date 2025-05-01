ARG ROS_DISTRO=melodic

FROM osrf/ros:${ROS_DISTRO}-desktop-full
LABEL authors="joshua"
SHELL ["/bin/bash", "--login", "-c"]

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3-pip \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/

# Install jackal gazebo simulation
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-jackal-simulator \
    ros-${ROS_DISTRO}-jackal-desktop \
    ros-${ROS_DISTRO}-jackal-navigation \
    ros-${ROS_DISTRO}-jackal-viz \
    && rm -rf /var/lib/apt/lists/

# Install cartographer dependencies
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-gazebo-plugins \
    ros-${ROS_DISTRO}-cartographer \
    ros-${ROS_DISTRO}-cartographer-ros \
    && rm -rf /var/lib/apt/lists/

# Install python dependencies
RUN pip3 install --no-cache-dir \
    defusedxml \
    netifaces \
    numpy \
    PyYAML \
    rospkg

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /catkin_ws

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
