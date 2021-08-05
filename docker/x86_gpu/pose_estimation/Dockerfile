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

# Fix an issue with the ROS key - https://askubuntu.com/questions/1341378/invalid-signature-error-for-ros-repository-while-trying-to-do-sudo-apt-get-updat
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654

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
    gdown https://drive.google.com/uc?id=1eIuFLNKjBT1zu3D6wNklrQ1PJ31N-ScV && \
    cd ~/msl_raptor_ws/src/msl_raptor/src/front_end/yolov3/cfg && \
    gdown https://drive.google.com/uc?id=1k8SvfBfU3YZk-Ogb8VjJ8OWcTkFfL0ab && \
    gdown https://drive.google.com/uc?id=10F2IS2qRmOuAbkooLeY58-wpJ-tgEKvh && \
    pip3 uninstall -y gdown && \
    rm -rf /var/lib/apt/lists/*

# Torch to TensorRT package
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install
    
RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Install edgetpu detection stuff ############################
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6A030B21BA07F4FB && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 8B57C5C2836F4BEB 

RUN apt update && apt install -y \
        libedgetpu1-std \
        python3-pycoral \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt update

# https://www.tensorflow.org/lite/guide/python  <-- different version for ARM
RUN pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp36-cp36m-linux_x86_64.whl

RUN mkdir /root/msl_raptor_ws/src/msl_raptor/src/front_end/coral && cd /root/msl_raptor_ws/src/msl_raptor/src/front_end/coral && \
    git clone https://github.com/google-coral/tflite.git && \
    cd /root/msl_raptor_ws/src/msl_raptor/src/front_end/coral/tflite/python/examples/detection && \
    bash install_requirements.sh
# end edgetpu stuff #####################################

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

RUN cd /root/msl_raptor_ws/src/msl_raptor/src/front_end/coral/tflite/python/examples/detection && \
    cp detect.py detect_coral.py && \
    cp detect_image.py detect_image_coral.py

# Add aliases to bashrc in docker containers
RUN echo 'alias raptor_go="clear && roslaunch msl_raptor msl_raptor.launch"' >> ~/.bashrc && \
    echo 'alias viz_go="roslaunch msl_raptor raptor_viz.launch"' >> ~/.bashrc && \
    echo 'alias logs_go="python3 ~/msl_raptor_ws/src/msl_raptor/src/utils_msl_raptor/rosbag_to_logs.py"' >> ~/.bashrc && \
    echo 'alias metrics_go="python3 ~/msl_raptor_ws/src/msl_raptor/src/viz_tools/result_analyser.py"' >> ~/.bashrc && \
    echo 'alias weight_reset_go="rm ~/msl_raptor_ws/src/msl_raptor/src/front_end/SiamMask/weights_trt/*.pth"' >> ~/.bashrc && \
    echo 'alias detection_go="cd /root/msl_raptor_ws/src/msl_raptor/src/front_end/coral/tflite/python/examples/detection && python3 detect_image.py --model /mounted_folder/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite --labels /mounted_folder/models/coco_labels.txt --input images/grace_hopper.bmp"' >> ~/.bashrc && \
    echo 'alias rosbag_detect_go="python3 ~/msl_raptor_ws/src/msl_raptor/src/utils_msl_raptor/detect_objects_rosbag.py"' >> ~/.bashrc

RUN source /ros_python3_entrypoint.sh && \
    cd msl_raptor_ws/src/msl_raptor && \
    git pull --recurse-submodules && \
    git checkout detect_only && \
    cd ../ocam_msl && \
    git pull && \
    catkin build

# Add git aliases for convience
RUN git config --global alias.co checkout && \
    git config --global alias.br branch && \
    git config --global alias.ci commit && \
    git config --global alias.st status
    
# Set GIT config based on local env vars
ARG GIT_NAME=
ARG GIT_EMAIL=
RUN git config --global user.email "$GIT_EMAIL" && \
    git config --global user.name "$GIT_NAME"

COPY ./msl_raptor_entrypoint.sh /

ENTRYPOINT ["/msl_raptor_entrypoint.sh"]
CMD ["bash"]
