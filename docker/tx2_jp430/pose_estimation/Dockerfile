FROM bramtoula/ros:melodic-trt-py3-tx2-jp430
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

# Scipy
RUN apt-get update && \
    apt-get install -y libatlas-base-dev gfortran && \
    pip3 install wheel && \
    pip3 install pybind11 && \
    pip3 install scipy && \
    rm -rf /var/lib/apt/lists/* 

# Pytorch
RUN apt-get update && \
    pip3 install numpy==1.16 && \
    apt-get install -y python3-pip libatlas-base-dev git libjpeg-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/* 

RUN wget https://nvidia.box.com/shared/static/phqe92v26cbhqjohwtvxorrwnmrnfx1o.whl -O torch-1.3.0-cp36-cp36m-linux_aarch64.whl && \
    pip3 install torch-1.3.0-cp36-cp36m-linux_aarch64.whl && \
    git clone --branch v0.4.2 https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    python3 setup.py install && \
    cd .. && \
    rm torch-1.3.0-cp36-cp36m-linux_aarch64.whl && \
    rm -rf torchvision

# SiamMask
RUN cd msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask && \
    apt-get update && \
    apt-get install -y python3-h5py python3-llvmlite python3-pandas python3-numba && \
    pip3 install Cython==0.29.4 colorama==0.3.9 requests==2.21.0 fire==0.1.3 tqdm==4.29.1 tensorboardX==1.6 && \
    /bin/bash make.sh && \
    cd experiments/siammask_sharp && \
    # Download weights
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth  && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth && \
     # Torch2trt weights
    pip3 install gdown && \
    mkdir ~/msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask/weights_trt && \
    cd ~/msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask/weights_trt && \
    gdown https://drive.google.com/uc?id=1tpJMg-A9BYPpbVlVeDPrpihhvXaTEeDn && \
    pip3 uninstall -y gdown && \
    tar -xzvf siammask_trt_weights.tar.gz && \
    rm siammask_trt_weights.tar.gz  && \
    rm -rf /var/lib/apt/lists/* 

# Set up YOLO
RUN apt-get update && \
    apt-get install -y python3-tk && \
    touch msl_raptor_ws/src/msl_raptor/src/front_end/yolov3/__init__.py && \
    # Download weights
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
    pip3 install pillow==6.2 && \
    python3 setup.py install && \
    cd .. && \
    rm -rf torch2trt
    
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

# Modify avahi-config (this has to be done)
RUN perl -p -i -e 's|#deny-interfaces=eth1|deny-interfaces=eth0,eth1|g' /etc/avahi/avahi-daemon.conf

# ---------------------------- Reset argument ------------------------------ #
ARG rebuild=0
# -------------------------------------------------------------------------- #

#############   DEBUG TOOLS   ################
# Install avahi debug tools such as avahi-resolve (good for testing, not needed for running)
RUN apt-get update && \
    apt-get install -y avahi-utils && \
    rm -rf /var/lib/apt/lists/*
##############################################

# Add aliases to bashrc in docker containers
RUN echo 'alias raptor_go="clear && roslaunch msl_raptor msl_raptor.launch"' >> ~/.bashrc && \
    echo 'alias viz_go="roslaunch msl_raptor raptor_viz.launch"' >> ~/.bashrc && \
    echo 'alias logs_go="python3 ~/msl_raptor_ws/src/msl_raptor/src/utils_msl_raptor/rosbag_to_logs.py"' >> ~/.bashrc && \
    echo 'alias metrics_go="python3 ~/msl_raptor_ws/src/msl_raptor/src/viz_tools/result_analyser.py"' >> ~/.bashrc
    
RUN source /ros_python3_entrypoint.sh && \
    cd msl_raptor_ws/src/msl_raptor && \
    git pull --recurse-submodules && \
    catkin build

ARG GIT_NAME=
ARG GIT_EMAIL=
RUN git config --global user.email "$GIT_EMAIL" && \
    git config --global user.name "$GIT_NAME"

COPY ./msl_raptor_entrypoint.sh /

ENTRYPOINT ["/msl_raptor_entrypoint.sh"]
CMD ["bash"]
