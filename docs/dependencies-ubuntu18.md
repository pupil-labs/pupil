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

## Continue Installation

That's it! You are done! Continue to follow the rest of the instructions on the [project main page](..).
