# Docker Tricks and Tips

## 0) tl;dr - The following are the build & run commands I used.
To build docker image, replacing <arch> with your architecture:
> `build_raptor_go='cd <workspace path>/docker/<arch>/pose_estimation && docker build . -t msl_raptor --build-arg GIT_NAME=$DOCKER_GIT_NAME --build-arg GIT_EMAIL=$DOCKER_GIT_EMAIL'`
- the --build-arg gave me access to variables in the Dockerfile, which I used as a way to pass through certain environmental variables. In this case the I used passed in my local env vars to configure git inside the container.

Before running the container, I created a .env file located at `<workspace path>/docker/`. To start the container:

> `docker run --rm -it --gpus all --privileged --env-file <workspace path>/docker/.env -v <env path>/msl_raptor/data:/bags:ro -env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --network host msl_raptor:latest`
- --gpus all gave me access to gpus
- --privileged gave me access to usb devices
- --env-file <workspace path>/docker/.env allowed me to set env variables in the container
- --network host allowed me to access the network my computer was connected to from within the container
- -v <path> mounted a folder from my computer in the docker container, so files could be accessed from both. The :ro at the end makes these files read only in the container.
- -env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" allowed VS code to display (very similar to ssh w/ -X option)
- To force a full rebuild of the image include `--no-cache` at the end of the docker build command.

- To force a rebuild of only part of the Dockerfile, include a commandline-settable argument and simply change its value in the build command using `--build-arg <build arg>=<random value>`. 
    - For example in Dockerfile: `ARG rebuild=0`, then call `docker build... --build-arg rebuild=42`.

## 1) Getting Started - Using Docker with Visual Studio Code
For my setup I used VS Code with Docker. The Remote-Contienrs extension made it easy to edit files inside the container, and the git extensions were also useful. I followed the standard installation instructions for Docker, and since I was using a GPU I also installed nvidea-docker.

## 2) Setting Env Variables
### Method A) in the `docker build` command
The following is is good method if you just need one or two variables set whos values will not change often. These variables will also be accesable within the Dockerfile.

To set the env var `MY_ENV_VAR` to [env_value] include the following in the build command:
`--build-arg MY_ENV_VAR=[env_value]`

The downside of this method is you have to rebuild the image if you want to change the value of the variable. See method B for a way to set the variables with the run command.

#### Example: Setting git.config user.name and user.email
In my .bashrc I exported two local environmental variables
>`export DOCKER_GIT_EMAIL="adam.w.caccavale@gmail.com"`               
>`export DOCKER_GIT_NAME="Adam"`

Now when I build my docker image I use
>`--build-arg GIT_NAME=$DOCKER_GIT_NAME --build-arg GIT_EMAIL=$DOCKER_GIT_EMAIL`

The full command for building `msl_raptor` is then:
>`docker build . -t msl_raptor --build-arg GIT_NAME=$DOCKER_GIT_NAME --build-arg GIT_EMAIL=$DOCKER_GIT_EMAIL`

These variables are now accessable in my Dockerfile, and I use the following command to configure my git settings.

> `RUN git config --global user.email "$GIT_EMAIL" && git config --global user.name "$GIT_NAME"`

### Method B) in the `docker run' command
For this method you need to create a file with the env variabls defined and use the `--env-file` option. The file can be named anything and here we use `.env`. This file contains definitions of the form: `ENV_VAR=ENV_VAR_VALUE`
>`--env-file [path to file]/.env`

## 3) Using ROS with the container
Including the run option `--network host` means "container’s network stack is not isolated from the Docker host (the container shares the host’s networking namespace), and the container does not get its own IP-address allocated". This was useful when we wanted to be able to communicate over ROS to another computer.  

The next step was to set the ROS_MASTER_URI and ROS_HOSTNAME environemental variables. These are already exported in my .bashrc, but to get them into the docker container they need to also be in the .env file. 

This file can be created manually, but I also created a bash script that creates this file for me. I assume the user is using avahi_daemon. Since the computer running the roscore's hostname is relay, my `ROS_MASTER_URI` will be http://relay.local:11311. However, for some reason we could not get avahi-daemon to start on its own, and had to launch it manually. 

#### 3a. Script to Create the .env File
This assumes the variables `ROS_HOSTNAME` and `ROS_MASTER_URI` are set. I also set a variable that is the path to my docker workspase called `DOCKER_PATH`.

> `docker_env_go() { 
    echo 'ROS_HOSTNAME='${ROS_HOSTNAME} > ${DOCKER_PATH}/docker/.env;
    echo 'ROS_MASTER_URI='${ROS_MASTER_URI} >> ${DOCKER_PATH}/docker/.env; }`

#### 3b. Getting Avahi to Work

The first thing I did was add a line to edit the avahi config file so it would only "look" for the roscore over the correct network adapter. I ran ifconfig to see the adapters, and I chose the name of the one I was using to connect to ROS. For example, I was using ethernet for internet and wireless for ROS and my wifi network adapter is called wlan0. I therefore add the following line to my Dockerfile. Make sure to replace wlan0 with the name of your network adapter.

> `RUN perl -p -i -e 's|#deny-interfaces=eth1|deny-interfaces=wlan0|g' /etc/avahi/avahi-daemon.conf`

At first avahi refused to start even when the command was in the enterypoint. My frist solution was to add an alias to run at the start of entering the container. To do that I added this line to my Dockerfile:

>`RUN echo 'alias avahi_go="/etc/init.d/dbus start && service avahi-daemon start"' >> ~/.bashrc`

However, eventually I found that starting avahi earlier in the entrypoint (before I sourced nvidea's entrypoint) fixed this issue. The lines I added to the entrypoint are the same as from the alias. Note that `&>/dev/null' suppresses the output of these lines.

>`/etc/init.d/dbus start &>/dev/null`

>`service avahi-daemon start &>/dev/null`


## 4.0 Dockerfile Stuff

For a bunch of best pratices for dockerfiles, see the following: https://docs.docker.com/develop/develop-images/dockerfile_best-practices/

Also note that trying to source a ros setup.bash or an entrypoint file will fail using `RUN source /my_entrypoint.sh`. This has to do with how Docker handles shell commands, and can be fixed by included the following line at the top of the file:

> `SHELL ["/bin/bash", "-c"] ` 

Some answers online say to use something like `RUN /bin/bash -c "source /my_entrypoint.sh"`, but this will cause other subtle issues.

(sources: https://stackoverflow.com/questions/20635472/using-the-run-instruction-in-a-dockerfile-with-source-does-not-work/39777387#39777387
and 
https://stackoverflow.com/questions/20635472/using-the-run-instruction-in-a-dockerfile-with-source-does-not-work)


## 5.0 Using GUIs with Docker
This is helpful: https://wiki.ros.org/docker/Tutorials/GUI, especially the following lines:

>`xhost +local:root`

and its opposite...

> `xhost -local:root`

Its also useful to add the env variable `DISPLAY` to docker using the previously described methods.

#### Getting Rviz Working
Even when other gui programs (like xclock) are working, rviz was still failing. This had to do with hardware acceleration not being enabled. 

First, I had to install `nvidia-container-runtime` (https://github.com/nvidia/nvidia-container-runtime#docker-engine-setup)and `nvidia-docker2` (sudo apt-get install nvidia-docker2).

Then I added two env variables to my Dockerfile and a few more arguments to my build command. 

Build args:
> `-env="XAUTHORITY=$XAUTH" --volume="$XAUTH:$XAUTH" --runtime=nvidia`

Dockerfile env vars:
> `ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}`

> `ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics`

<!-- 
NOTE: THIS FIX ISNT VERIFIED YET
#### Getting PyPlot Working

You need to use a different backend for matplotlib than the default to use it in a virtual container or else you get an error like `RuntimeError: main thread is not in main loop`. To do this install a few things:

>`apt-get install tcl-dev tk-dev python-tk python3-tk`

and then (BEFORE YOU IMPORT PYPLOT) include the following lines:

>`import matplotlib`

>`matplotlib.using('TkAgg')`

sourse: https://www.pyimagesearch.com/2015/08/24/resolved-matplotlib-figures-not-showing-up-or-displaying/ -->