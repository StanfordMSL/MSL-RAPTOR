# Docker Tricks and Tips

## 1. Using Docker with Visual Studio Code
to do

## 2. Setting Env Variables
### Method A) in the `docker build` command
This is good if you just need one or two, and they will not change values often.

If you want to sent the env var `MY_ENV_VAR` to [env_value] include the following in the build command:
`--build-arg MY_ENV_VAR=[env_value]`

#### Example: Setting git.config user.name and user.email
In my .bashrc I exported two local environmental variables
>`export DOCKER_GIT_EMAIL="adam.w.caccavale@gmail.com"`
>
>`export DOCKER_GIT_NAME="Adam"`

Now when I build my docker image I use
>`--build-arg GIT_NAME=$DOCKER_GIT_NAME --build-arg GIT_EMAIL=$DOCKER_GIT_EMAIL`

The full command for building `msl_raptor` is then:
>`docker build . -t msl_raptor --build-arg GIT_NAME=$DOCKER_GIT_NAME --build-arg GIT_EMAIL=$DOCKER_GIT_EMAIL`

Note you can include `--no-cache` at the end of this command to force a rebuild of the entire image

### Method B) in the `docker run' command
For this method you need to create a file with the env variabls defined and use the `--env-file` option. The file can be named anything and here we use `.env`. This file contains definitions of the form: `ENV_VAR=ENV_VAR_VALUE`
>`--env-file [path to file]/.env`


## 3. Using ROS with the container
Including the run option `--network host` means "container’s network stack is not isolated from the Docker host (the container shares the host’s networking namespace), and the container does not get its own IP-address allocated". This was useful when we wanted to be able to communicate over ros to another computer.  

The next step was to set the ROS_MASTER_URI and ROS_HOSTNAME environemental variables. These are already exported in my .bashrc, but to get them into the docker container they need to also be in the .env file. 

This file can be created manually, but I also created a bash script that creates this file for me. I assume the user is using avahi_daemon. Since the computer running the roscore's hostname is relay, my `ROS_MASTER_URI` will be http://relay.local:11311. However, for some reason we could not get avahi-daemon to start on its own, and had to launch it manually. 

#### 3a. Script to Create the .env File
This assumes the variables `ROS_HOSTNAME` and `ROS_MASTER_URI` are set.

> `docker_env_go() { 
    echo 'ROS_HOSTNAME='${ROS_HOSTNAME} > /home/adam/Documents/msl_raptor_ws/src/msl_raptor/docker/.env;
    echo 'ROS_MASTER_URI='${ROS_MASTER_URI} >> /home/adam/Documents/msl_raptor_ws/src/msl_raptor/docker/.env; }`

#### 3b. Getting Avahi to Work

The first thing I did was add a line to edit the avahi config file so it would only "look" for the roscore over the correct network adapter. I ran ifconfig to see the adapters, and I chose the name of the one I was using to connect to ROS. For example, I was using ethernet for internet and wireless for ROS and my wifi network adapter is called wlan0. I therefore add the following line to my Dockerfile. Make sure to replace wlan0 with the name of your network adapter.

> `RUN perl -p -i -e 's|#deny-interfaces=eth1|deny-interfaces=wlan0|g' /etc/avahi/avahi-daemon.conf`

Since avahi refused to start even when the command was in the Dockerfile and/or the enterypoint, I added an alias to run at the start of entering the container. To do that I added this line to my Dockerfile:

>`RUN echo 'alias avahi_go="/etc/init.d/dbus start && service avahi-daemon start"' >> ~/.bashrc`


## logging into the tx2

ssh (thorugh vs code insdier?)
cd Documents/MSL-RAPTOR
git pull
use vs code to run image interactive



for camera: 
needed sudo apt-get install libudev-dev


