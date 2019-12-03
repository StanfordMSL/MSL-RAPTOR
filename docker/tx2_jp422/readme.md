On the TX2 after flashing (tested on Jetpack 4.2.2), copy the csv file to have more libraries added with the nvidia runtime:
```
sudo cp python_libs.csv /etc/nvidia-container-runtime/host-files-for-container.d/
```

Make the nvidia runtime default by adding `"default-runtime": "nvidia"` to `/etc/docker/daemon.json` (https://github.com/NVIDIA/nvidia-container-runtime#docker-engine-setup).

Make sure the numpy version installed in `/usr/lib/python3/dist-packages/numpy` is at least 1.16.
You can install a more recent version with `pip3 install --upgrade numpy`, and check where it is installed with `pip3 show numpy`. 
You can move it to the right place if needed.


You can then build this image, and run it with the nvidia runtime.
