FROM bramtoula/ros:melodic-python3


# Install sublime
RUN apt-get update &&\
    apt install -y apt-transport-https ca-certificates curl software-properties-common && \
    curl -fsSL https://download.sublimetext.com/sublimehq-pub.gpg | apt-key add - && \
    add-apt-repository "deb https://download.sublimetext.com/ apt/stable/" && \
    apt-get update && \
    apt install -y sublime-text


# Clone and build MSL-RAPTOR with python3
RUN source /ros_entrypoint.sh && source /ros_python3_entrypoint.sh && \
    mkdir ~/msl_raptor_ws && cd ~/msl_raptor_ws && \
    catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so && \
    catkin config --install && \
    mkdir ~/msl_raptor_ws/src && \
    cd  ~/msl_raptor_ws/src && \
    git clone https://github.com/StanfordMSL/MSL-RAPTOR.git && \
    cd ~/msl_raptor_ws && \
    catkin build

    
# ----------------------------- Reset argument --------------------------------#
ARG rebuild=0
# -----------------------------------------------------------------------------#
RUN cd ~/msl_raptor_ws/src/msl_raptor && \
    git pull && \
    catkin build