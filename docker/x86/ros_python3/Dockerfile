FROM ros:melodic-perception

# Basic stuff for Python 3 with ROS
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip python-catkin-tools python3-dev python3-numpy python3-yaml && \
    pip3 install rospkg catkin_pkg && \
    rm -rf /var/lib/apt/lists/* 

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Rebuild cv_bridge with python3
RUN source /ros_entrypoint.sh && \
    mkdir ~/cv_bridge_py3_ws && cd ~/cv_bridge_py3_ws && \
    catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so && \
    catkin config --install && \
    mkdir ~/cv_bridge_py3_ws/src && \
    cd  ~/cv_bridge_py3_ws/src && \
    git clone -b melodic https://github.com/ros-perception/vision_opencv.git && \
    cd ~/cv_bridge_py3_ws && \
    catkin build cv_bridge && \
    source install/setup.bash --extend

COPY ./ros_python3_entrypoint.sh /

ENTRYPOINT ["/ros_python3_entrypoint.sh"]
CMD ["bash"]
