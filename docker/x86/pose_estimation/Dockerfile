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
    catkin build

# Set up SiamMask
RUN cd msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask && \
    pip3 install -r requirements.txt && \
    /bin/bash make.sh && \
    cd experiments/siammask_sharp && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth  && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth && \
    pip3 install scipy --upgrade

# Set up YOLO
RUN apt-get update && \
    apt-get install -y python3-tk && \
    touch msl_raptor_ws/src/msl_raptor/src/front_end/yolov3/__init__.py && \
    # Download drone detection weights
    pip3 install gdown && \
    cd ~/msl_raptor_ws/src/msl_raptor/src/front_end/yolov3/weights && \
    gdown https://drive.google.com/uc?id=1UYu8E8ZmwSz2xkTCBYq44vOAn28fjdXa && \
    cd ~/msl_raptor_ws/src/msl_raptor/src/front_end/yolov3/cfg && \
    gdown https://drive.google.com/uc?id=140b6leURXKktVPk1qGG5IvsHYGq2mGos && \
    pip3 uninstall -y gdown && \
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

# Make sure avahi-daemon is running and avahi-resolve is installed
RUN apt-get update && \
    apt-get install -y avahi-utils 
    # /etc/init.d/dbus start && \
    # service avahi-daemon start

# Add aliases to bashrc in docker containers
RUN echo 'alias raptor_go="clear && roslaunch msl_raptor msl_raptor.launch b_rosbag:=true"' >> ~/.bashrc
RUN echo 'alias avahi_go="/etc/init.d/dbus start && service avahi-daemon start"' >> ~/.bashrc

RUN source /ros_python3_entrypoint.sh && \
    cd msl_raptor_ws/src/msl_raptor && \
    git pull && \
    catkin build

# Set GIT config based on local env vars
ARG GIT_NAME=
ARG GIT_EMAIL=
RUN git config --global user.email "$GIT_EMAIL" && \
    git config --global user.name "$GIT_NAME"

COPY ./msl_raptor_entrypoint.sh /

ENTRYPOINT ["/msl_raptor_entrypoint.sh"]
CMD ["bash"]
