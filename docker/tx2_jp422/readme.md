On the TX2 after flashing (tested on Jetpack 4.2.2), copy the csv file to have more libraries added with the nvidia runtime:
```
sudo cp python_libs.csv /etc/nvidia-container-runtime/host-files-for-container.d/
```

Make the nvidia runtime default by adding `"default-runtime": "nvidia"` to `/etc/docker/daemon.json` (https://github.com/NVIDIA/nvidia-container-runtime#docker-engine-setup).

You can then build this image, and run it with the nvidia runtime.
