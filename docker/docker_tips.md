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



#### Avahi-daemon workaround - using .bashrc functions to generate an .env file with ros environmental variables
In our lab we use avahi-daemon to resolve host names (so we do not have static ips). This is a problem because we were unable to resolve the hostnames in the container. If this is not a problem for you, skip this section.

These lines are in my .bashrc. The function could be done in a single line, but for clarity I break it up into three. The first task is to get our own IP address. Run ifconfig on your computer, and look for the network adapter name you want (i.e. wlp2s0).
>`get_ip_addr() { ip addr list wlp2s0 | awk -F'[ /]+' '$2=="inet"{print $3}'; }`

The next line uses avahi daemon to resolve the hostname, ours is `relay.local`. The -4 is to force it to use ip4 addres, if you want ip6 you can use -6 or leave blank to let it choose.
>`resolve_relay_hs() { $(avahi-resolve -n -4 relay.local); }`

Finally, this last function uses the previous two. The first part overwrites the existing .env file with the ROS_HOSTNAME variable. The next appends to it with the ROS_MASTER_URI.
>`create_env() { echo 'ROS_HOSTNAME='$(get_ip_addr) > [path_to_env_file]/.env && echo 'ROS_MASTER_URI=http://'$(resolve_relay_hs)':11311' >> [path_to_env_file]/.env; }`

