On the TX2 after flashing (tested on Jetpack 4.3.0), copy the csv file to have more libraries added with the nvidia runtime:
```
sudo cp python_libs.csv /etc/nvidia-container-runtime/host-files-for-container.d/
```

Make the nvidia runtime default by adding `"default-runtime": "nvidia"` to `/etc/docker/daemon.json` (https://github.com/NVIDIA/nvidia-container-runtime#docker-engine-setup). This has to be done to build the pose_estimation Dockerfile using the correct libraries. However it might cause issue if you rebuild the ros_python3 and ros_tensorrt ones, which you should do without the nvidia runtime.

You can then build this image, and run it with the nvidia runtime.
