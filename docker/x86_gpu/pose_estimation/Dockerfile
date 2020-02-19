FROM bramtoula/ros:melodic-trt-py3
WORKDIR /root

# Change default shell for RUN per https://stackoverflow.com/questions/20635472/using-the-run-instruction-in-a-dockerfile-with-source-does-not-work/39777387#39777387
SHELL ["/bin/bash", "-c"]

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
    pip3 install scipy --upgrade && \
    # Torch2trt weights
    pip3 install gdown && \
    mkdir ~/msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask/weights_trt && \
    cd ~/msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask/weights_trt && \
    gdown https://drive.google.com/uc?id=1_VR2c2UpW8UoZz5k4LS835jFj_7K2bAR && \
    pip3 uninstall -y gdown && \
    tar -xzvf siammask_trt_weights.tar.gz && \
    rm siammask_trt_weights.tar.gz

# Set up YOLO
RUN apt-get update && \
    apt-get install -y python3-tk && \
    touch msl_raptor_ws/src/msl_raptor/src/front_end/yolov3/__init__.py && \
    # Download drone detection weights
    pip3 install gdown && \
    cd ~/msl_raptor_ws/src/msl_raptor/src/front_end/yolov3/weights && \
    gdown https://drive.google.com/uc?id=1ZkK8AwVH1_MWo4tP6L4uGl9NxCf_SzrG && \
    cd ~/msl_raptor_ws/src/msl_raptor/src/front_end/yolov3/cfg && \
    gdown https://drive.google.com/uc?id=1k8SvfBfU3YZk-Ogb8VjJ8OWcTkFfL0ab && \
    pip3 uninstall -y gdown && \
    rm -rf /var/lib/apt/lists/*

# Torch to TensorRT package
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install
    
RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Clone OCAM github (and install dependences) --> this allows us to use our camera
RUN apt-get update && \
    apt-get install -y libudev-dev && apt-get install -y libv4l-dev && \
    cd ~/msl_raptor_ws/src/ && \
    git clone https://github.com/StanfordMSL/ocam_msl.git && \
    cd ~/msl_raptor_ws/ && \
    catkin build && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies for rosbag (needed for reading data from a rosbag)
RUN pip3 install pycrypto python-gnupg

# Modify avahi-config
RUN perl -p -i -e 's|#allow-interfaces=eth0|allow-interfaces=wlp2s0,wlp2s1|g' /etc/avahi/avahi-daemon.conf

# ---------------------------- Reset argument ------------------------------ #
ARG rebuild=0
# -------------------------------------------------------------------------- #

#############   DEBUG TOOLS   ################
# Install avahi debug tools such as avahi-resolve (good for testing, not needed for running)
RUN apt-get update && \
    apt-get install -y avahi-utils && \
    rm -rf /var/lib/apt/lists/*

# Install 'ping'
RUN apt-get update && \
    apt-get install -y iputils-ping && \
    rm -rf /var/lib/apt/lists/*
##############################################

# Add aliases to bashrc in docker containers
RUN echo 'alias raptor_go="clear && roslaunch msl_raptor msl_raptor.launch"' >> ~/.bashrc && \
    echo 'alias viz_go="roslaunch msl_raptor raptor_viz.launch"' >> ~/.bashrc && \
    echo 'alias logs_go="python3 ~/msl_raptor_ws/src/msl_raptor/src/utils_msl_raptor/rosbag_to_logs.py"' >> ~/.bashrc && \
    echo 'alias metrics_go="python3 ~/msl_raptor_ws/src/msl_raptor/src/viz_tools/result_analyser.py"' >> ~/.bashrc && \
    echo 'alias weight_reset_go="rm ~/msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask/weights_trt/*.pth"' >> ~/.bashrc

RUN source /ros_python3_entrypoint.sh && \
    cd msl_raptor_ws/src/msl_raptor && \
    git pull --recurse-submodules && \
    catkin build

# Set GIT config based on local env vars
ARG GIT_NAME=
ARG GIT_EMAIL=
RUN git config --global user.email "$GIT_EMAIL" && \
    git config --global user.name "$GIT_NAME"

COPY ./msl_raptor_entrypoint.sh /

ENTRYPOINT ["/msl_raptor_entrypoint.sh"]
CMD ["bash"]
