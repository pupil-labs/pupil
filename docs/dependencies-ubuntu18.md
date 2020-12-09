# Dependencies for Ubuntu 18.04 LTS

This setup is only tested on Ubuntu 18.04 LTS. Do not run Pupil on a VM unless you know what you are doing. For setting up your dependencies on Ubuntu 17.10 or lower, take a loot at [the corresponding setup guide](./dependencies-ubuntu17.md).

Most of this works via **apt**! Just copy paste into the terminal and listen to your machine purr.

## General Dependencies

```sh
sudo apt install -y pkg-config git cmake build-essential nasm wget python3-setuptools libusb-1.0-0-dev  python3-dev python3-pip python3-numpy python3-scipy libglew-dev libtbb-dev

# ffmpeg >= 3.2
sudo apt install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev ffmpeg x264 x265 libportaudio2 portaudio19-dev

# OpenCV >= 3 + Eigen
sudo apt install -y python3-opencv libopencv-dev libeigen3-dev
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

## Custom Version of libusb for 200hz cameras

This is **ONLY** required if you are using 200hz cameras. Otherwise it can be ignored!

1. Build or download fixed binary from release: https://github.com/pupil-labs/libusb/releases/tag/v1.0.21-rc6-fixes
1. Replace system libusb-1.0.so.0 with this binary. You can find the path of the system library with

    ```sh
    dpkg -L libusb-1.0-0-dev | grep libusb-1.0.so
    ```

    Please note that this command gives you the location of `libusb-1.0.so` while you need to replace `libusb-1.0.so.0`, but the required file should be found in the same folder.


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

### Install Python Libraries

We recommend using a [virtual environment](https://docs.python.org/3/tutorial/venv.html) for running Pupil. To install all Python dependencies, you can use the [`requirements.txt`](https://github.com/pupil-labs/pupil/blob/master/requirements.txt) file from the root of the `pupil` repository.

```sh
# Upgrade pip to latest version. This is necessary for some dependencies.
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

### OpenCV Troubleshooting
`ImportError: No module named 'cv2'`
  
When you see this error, Python cannot find the bindings from your OpenCV installation.

**We do NOT (!) recommend to install `opencv-python` via pip in that case!** 

Installing `opencv-python` will install another full (potentially different) version of opencv to your machine, so we are not recommending this setup.
When you install opencv with `sudo apt install -y python3-opencv libopencv-dev` as we advise above, you should have the `cv2` package available for import in Python as this will install compatible Python bindings already.

However, you might run into these problems when using a virtual environment, as your virtual environment cannot by default access Python packages that were installed from `apt`.
In that case there are 2 options:

1. Symlink or copy the Python bindings into your virtualenv. See e.g. [step 4 of this stackoverflow post](https://stackoverflow.com/a/37190408) for reference.
2. Create your virtualenv with the [`--system-site-packages`](https://virtualenv.pypa.io/en/latest/userguide/#the-system-site-packages-option) option, which will enable access to system-installed Python packages.

## Next Steps

That's it! You're done installing dependencies!

Now, you should be able to run Pupil from source. Follow the remaining instructions in the [README](../README.md).
