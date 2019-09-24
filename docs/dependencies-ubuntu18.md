# Dependencies for Ubuntu 18.04 LTS

This setup is only tested on Ubuntu 18.04 LTS. Do not run Pupil on a VM unless you know what you are doing. For setting up your dependencies on Ubuntu 17.10 or lower, take a loot at [the corresponding setup guide](./dependencies-ubuntu17.md).

Most of this works via **apt**! Just copy paste into the terminal and listen to your machine purr.

## General Dependencies

```sh
sudo apt install -y pkg-config git cmake build-essential nasm wget python3-setuptools libusb-1.0-0-dev  python3-dev python3-pip python3-numpy python3-scipy libglew-dev libglfw3-dev libtbb-dev

# ffmpeg >= 3.2
sudo apt install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev ffmpeg x264 x265 libportaudio2 portaudio19-dev

# OpenCV >= 3
sudo apt install -y python3-opencv libopencv-dev

# 3D Eye model dependencies
sudo apt install -y libboost-dev
sudo apt install -y libboost-python-dev
sudo apt install -y libgoogle-glog-dev libatlas-base-dev libeigen3-dev
sudo apt install -y libceres-dev
```

## Turbojpeg

```sh
wget -O libjpeg-turbo.tar.gz https://sourceforge.net/projects/libjpeg-turbo/files/1.5.1/libjpeg-turbo-1.5.1.tar.gz/download
tar xvzf libjpeg-turbo.tar.gz
cd libjpeg-turbo-1.5.1
./configure --enable-static=no --prefix=/usr/local
sudo make install
sudo ldconfig
```

## libuvc
```sh
git clone https://github.com/pupil-labs/libuvc
cd libuvc
mkdir build
cd build
cmake ..
make && sudo make install
```

If you want to run libuvc as normal user, add the following udev rules:
```sh
echo 'SUBSYSTEM=="usb",  ENV{DEVTYPE}=="usb_device", GROUP="plugdev", MODE="0664"' | sudo tee /etc/udev/rules.d/10-libuvc.rules > /dev/null
sudo udevadm trigger
```

## Python Libraries

```sh
sudo pip3 install numexpr
sudo pip3 install cython
sudo pip3 install psutil
sudo pip3 install pyzmq
sudo pip3 install msgpack==0.5.6
sudo pip3 install pyopengl
sudo pip3 install pyaudio
sudo pip3 install cysignals
sudo pip3 install git+https://github.com/zeromq/pyre

sudo pip3 install pupil_apriltags
sudo pip3 install git+https://github.com/pupil-labs/PyAV
sudo pip3 install git+https://github.com/pupil-labs/pyuvc
sudo pip3 install git+https://github.com/pupil-labs/pyndsi
sudo pip3 install git+https://github.com/pupil-labs/pyglui
sudo pip3 install git+https://github.com/pupil-labs/nslr
sudo pip3 install git+https://github.com/pupil-labs/nslr-hmm
```

## (Optional) PyTorch + CUDA and cuDNN

### Version 1: Without GPU acceleration

```bash
pip3 install pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
```

Some bleeding edge features require the deep learning library PyTorch.
Without GPU acceleration some of the features will probably not
run in real-time.


### Version 2: With GPU acceleration

```bash
pip3 install torch torchvision
```

Please refer to the following links on how to install CUDA and cuDNN:

- CUDA https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation
- cuDNN: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html