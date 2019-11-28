FROM bramtoula/ros:melodic-python3


# Install sublime
RUN apt-get update &&\
    apt install -y apt-transport-https ca-certificates curl software-properties-common && \
    curl -fsSL https://download.sublimetext.com/sublimehq-pub.gpg | apt-key add - && \
    add-apt-repository "deb https://download.sublimetext.com/ apt/stable/" && \
    apt-get update && \
    apt install -y sublime-text

# # Install Visual Studio Code
# RUN apt-get update &&\
#     apt install -y software-properties-common apt-transport-https wget && \
#     wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add - && \
#     add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" && \
#     apt-get update && \
#     apt install -y code && \
#     apt install -y libasound2 && \
#     mkdir /home/docker/code


# Clone and build MSL-RAPTOR with python3
RUN source /ros_python3_entrypoint.sh && \
    mkdir ~/msl_raptor_ws && cd ~/msl_raptor_ws && \
    #catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so && \
    #catkin config --install && \
    mkdir ~/msl_raptor_ws/src && \
    cd  ~/msl_raptor_ws/src && \
    git clone https://github.com/StanfordMSL/MSL-RAPTOR.git msl_raptor && \
    cd ~/msl_raptor_ws && \
    catkin build


# ---------------------------- Reset argument ------------------------------ #
ARG rebuild=0
# -------------------------------------------------------------------------- #

# Add aliases to bashrc in docker containers
RUN echo 'alias subl_go="cd /root/msl_raptor_ws/src/msl_raptor && subl /root/msl_raptor_ws/src/msl_raptor/src/"' >> ~/.bashrc && \
    echo 'alias rosbag_go="rosbag play -l /bag.bag"' >> ~/.bashrc && \
    echo 'alias raptor_go="roslaunch msl_raptor msl_raptor.launch"' >> ~/.bashrc


RUN source /ros_python3_entrypoint.sh && \
    cd ~/msl_raptor_ws/src/msl_raptor && \
    git pull && \
    catkin build

ARG GIT_NAME=
ARG GIT_EMAIL=
RUN git config --global user.email "$GIT_EMAIL" && \
    git config --global user.name "$GIT_NAME"

COPY ./msl_raptor_entrypoint.sh /

ENTRYPOINT ["/msl_raptor_entrypoint.sh"]
CMD ["bash"]
