FROM bramtoula/ros:melodic-trt-tx2-jp430

WORKDIR /root
# Basic stuff for Python 3 with ROS
RUN apt-get update && \
    apt-get install -y --no-install-recommends python-catkin-tools python3-pip python3-dev python3-numpy python3-yaml python3-setuptools libbullet-dev && \
    pip3 install rospkg catkin_pkg empy && \
    rm -rf /var/lib/apt/lists/* 

RUN rm /bin/sh && ln -s /bin/bash /bin/sh


# Remove packages we will rebuild
RUN apt-get -y remove ros-melodic-cv-bridge ros-melodic-tf* ros-melodic-eigen-conversions && \
    apt-get -y remove 

# Rebuild cv_bridge with python3
RUN source /ros_entrypoint.sh && \
    mkdir python3_ws && cd python3_ws && \
    catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so && \
    catkin config --install && \
    mkdir src && \
    cd src && \
    git clone -b melodic-devel https://github.com/ros/geometry.git && \
    git clone -b melodic https://github.com/ros-perception/vision_opencv.git && \
    git clone -b melodic-devel https://github.com/ros/geometry2.git  && \
    cd .. && \
    catkin build && \
    source install/setup.bash --extend

# Reinstall python stuff
RUN apt-get update && \
    apt-get install python3-opencv && \
    rm -rf /var/lib/apt/lists/* 



COPY ./ros_python3_entrypoint.sh /

ENTRYPOINT ["/ros_python3_entrypoint.sh"]
CMD ["bash"]
