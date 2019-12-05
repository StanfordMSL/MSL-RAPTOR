FROM bramtoula/ros:melodic-trt-py3
WORKDIR /root

# Clone and build MSL-RAPTOR with python3
RUN source /ros_python3_entrypoint.sh && \
    mkdir msl_raptor_ws && cd msl_raptor_ws && \
    #catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so && \
    #catkin config --install && \
    mkdir src && \
    cd  src && \
    git clone --recurse-submodules https://github.com/StanfordMSL/MSL-RAPTOR.git msl_raptor && \
    cd .. && \
    catkin build && \
    pip3 install pyquaternion

# Set up SiamMask
RUN cd msl_raptor_ws/src/msl_raptor/src/front-end/SiamMask && \
    pip3 install -r requirements.txt && \
    /bin/bash make.sh && \
    cd experiments/siammask_sharp && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth  && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth

# Set up YOLO
RUN apt-get update && \
    apt-get install -y python3-tk && \
    touch msl_raptor_ws/src/msl_raptor/src/front-end/yolov3/__init__.py && \
    rm -rf /var/lib/apt/lists/*

# Torch to TensorRT package
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install
    
RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Install 'ping'
RUN apt-get update && \
    apt-get install -y iputils-ping
    
# ---------------------------- Reset argument ------------------------------ #
ARG rebuild=0
# -------------------------------------------------------------------------- #

# Add aliases to bashrc in docker containers
RUN echo 'alias raptor_go="clear && roslaunch msl_raptor msl_raptor.launch b_rosbag:=true"' >> ~/.bashrc

RUN source /ros_python3_entrypoint.sh && \
    cd msl_raptor_ws/src/msl_raptor && \
    git pull && \
    catkin build

COPY ./msl_raptor_entrypoint.sh /

ENTRYPOINT ["/msl_raptor_entrypoint.sh"]
CMD ["bash"]
