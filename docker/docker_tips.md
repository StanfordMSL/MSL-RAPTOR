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

The next step was to set the ROS_MASTER_URI and ROS_HOSTNAME environemental variables. These are already exported in my .bashrc, but to get them into the docker container they need to also be in teh .env file. 

This got complicated because in our lab we use avahi-daemon to resolve host names (so we do not have static ips). For example, normally the ROS_MASTER_URI is http://relay.local:11311, but in the container we were unable to resolve relay.local into an ip address. If this is not a problem for you, just follow the steps in Section 2 how to set an environemntal variable.


#### Avahi-daemon workaround - using .bashrc functions to generate an .env file with ros environmental variables

These lines are in my .bashrc. The function could be done in a single line, but for clarity I break it up into several. The first task is to get our own IP address. Run ifconfig on your computer, and look for the network adapter name you want (i.e. wlp2s0).
>`get_ip_addr() { ip addr list wlp2s0 | awk -F'[ /]+' '$2=="inet"{print $3}'; }`

I then parse my already-set env variable (ROS_MASTER_URI) to extract the host name of the roscore
>`parse_RMU_for_hs() { echo $ROS_MASTER_URI | awk -F':|/' '{print $4}'; }`

Next, I use avahi daemon to resolve the hostname (ours is `relay.local`). The -4 is to force it to use ip4 addres, if you want ip6 you can use -6 or leave blank to let it choose.
>`resolve_ros_hs() { echo $(avahi-resolve -n -4 $(parse_RMU_for_hs)) | awk '{ print $2 }'; }`

Finally, this last function uses the previous two. The first part overwrites the existing .env file with the ROS_HOSTNAME variable. The next appends to it with the ROS_MASTER_URI.
>`create_env() { echo 'ROS_HOSTNAME='$(get_ip_addr) > [path_to_env_file]/.env && echo 'ROS_MASTER_URI=http://'$(resolve_ros_hs)':11311' >> [path_to_env_file]/.env; }`



<FILL IN REST>

update avahi_go etc

## logging into the tx2

ssh (thorugh vs code insdier?)
cd Documents/MSL-RAPTOR
git pull
use vs code to run image interactive



for camera: 
needed sudo apt-get install libudev-dev