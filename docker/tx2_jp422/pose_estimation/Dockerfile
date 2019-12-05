FROM bramtoula/ros:melodic-trt-py3-tx2
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

# Pytorch
RUN apt-get update && \
    apt-get install -y python3-pip libatlas-base-dev git libjpeg-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/* 

RUN wget https://nvidia.box.com/shared/static/phqe92v26cbhqjohwtvxorrwnmrnfx1o.whl -O torch-1.3.0-cp36-cp36m-linux_aarch64.whl && \
    pip3 install torch-1.3.0-cp36-cp36m-linux_aarch64.whl && \
    git clone --branch v0.5.0 https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    python3 setup.py install

# SiamMask
RUN cd msl_raptor_ws/src/msl_raptor/src/front-end/SiamMask && \
    apt-get update && \
    apt-get install -y python3-h5py python3-scipy python3-llvmlite python3-pandas python3-numba && \
    pip3 install wheel && \
    pip3 install Cython==0.29.4 colorama==0.3.9 requests==2.21.0 fire==0.1.3 tqdm==4.29.1 tensorboardX==1.6 && \
    /bin/bash make.sh && \
    cd experiments/siammask_sharp && \
    # Download weights
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth  && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth && \
    rm -rf /var/lib/apt/lists/* 

# Set up YOLO
RUN apt-get update && \
    apt-get install -y python3-tk && \
    touch msl_raptor_ws/src/msl_raptor/src/front-end/yolov3/__init__.py && \
    # Download weights
    cd msl_raptor_ws/src/msl_raptor/src/front-end/yolov3/weights && \
    wget https://pjreddie.com/media/files/yolov3.weights && \
    wget https://pjreddie.com/media/files/yolov3-tiny.weights && \
    rm -rf /var/lib/apt/lists/*
# Torch to TensorRT package
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install
    
RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*
# ---------------------------- Reset argument ------------------------------ #
ARG rebuild=0
# -------------------------------------------------------------------------- #

# Add aliases to bashrc in docker containers
RUN echo 'alias rosbag_go="source /msl_raptor_entrypoint.sh && rosbag play -l /bag.bag"' >> ~/.bashrc && \
    echo 'alias raptor_go="source /msl_raptor_entrypoint.sh && roslaunch msl_raptor msl_raptor.launch"' >> ~/.bashrc

RUN source /ros_python3_entrypoint.sh && \
    cd msl_raptor_ws/src/msl_raptor && \
    git pull && \
    catkin build

ARG GIT_NAME=
ARG GIT_EMAIL=
RUN git config --global user.email "$GIT_EMAIL" && \
    git config --global user.name "$GIT_NAME"

COPY ./msl_raptor_entrypoint.sh /

ENTRYPOINT ["/msl_raptor_entrypoint.sh"]
CMD ["bash"]
