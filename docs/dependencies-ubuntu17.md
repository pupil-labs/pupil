# Dependencies for Ubuntu 17.10 or lower

These installation instructions are tested from **Ubuntu 16.04** to **Ubuntu 17.10** running on many machines. Do not run Pupil on a VM unless you know what you are doing. We recommend using **Ubuntu 18.04 LTS** though as the setup is much easier! See the corresponding setup guide for [Ubuntu 18.04](./dependencies-ubuntu18.md).


## General Setup

Pupil requires Python 3.6 or higher. Please check this [resource](https://askubuntu.com/questions/865554/how-do-i-install-python-3-6-using-apt-get) on how to install Python 3.6 on your version of Ubuntu.

```sh
sudo apt-get update
sudo apt install -y pkg-config git cmake build-essential nasm wget python3-setuptools libusb-1.0-0-dev  python3-dev python3-pip python3-numpy python3-scipy libglew-dev libtbb-dev
```

## ffmpeg3

Install ffmpeg3 from jonathonf's ppa:

```sh
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libavresample-dev ffmpeg libav-tools x264 x265 libportaudio2 portaudio19-dev
```

## OpenCV

You will need to install OpenCV from source:

**NOTE:** Opencv needs to following requirements set up correctly for being able to build correctly! If you don't have these set up correctly, you might be able to build OpenCV, but experience errors when trying to use it with Pupil!
1. **python3** interpreter can be found
2. __libpython***.so__ shared lib can be found (make sure to install python3-dev)
3. **numpy** for python3 is installed

If everything is set up correctly, run the following commands in the terminal:

```sh
git clone https://github.com/opencv/opencv
cd opencv
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DWITH_TBB=ON -DWITH_CUDA=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=ON ..
make -j2
sudo make install
sudo ldconfig
```

**NOTE:** OpenCV is not able to build Python 2 and Python 3 modules at the same time. OpenCV will build the Python 2 module by default if requirements for both versions are met. Setting the Python 2 Numpy include directory to an empty string effectively disables the Python 2 module build.

### OpenCV Troubleshooting

The following errors were commonly reported:

* `ImportError: No module named 'cv2'`

  When you see this error, Python cannot find the bindings from your OpenCV installation.
  
  **We do NOT (!) recommend to install `opencv-python` via pip in that case!** 
  
  Installing `opencv-python` will install another full (potentially different) version of opencv to your machine, so we are not recommending this setup.
  When you compile opencv with `-DBUILD_opencv_python3=ON` as we advise above, you should have the `cv2` package available for import in Python as this will install compatible Python bindings already.

  However, you might run into these problems when using a virtual environment, as your virtual environment cannot by default access Python packages that were installed from `apt`.
  In that case there are 2 options:
  
  1. Symlink or copy the Python bindings into your virtualenv. See e.g. [step 4 of this stackoverflow post](https://stackoverflow.com/a/37190408) for reference.
  2. Create your virtualenv with the [`--system-site-packages`](https://virtualenv.pypa.io/en/latest/userguide/#the-system-site-packages-option) option, which will enable access to system-installed Python packages.

  If you are still experiencing this issue, delete the OpenCV build folder, recheck the requirements and build and try again.


* `ImportError: */detector_2d.*.so: undefined symbol: *ellipse*InputOutputArray*RotatedRect*Scalar*`
  
  This error appears if opencv has been installed previously via **apt-get**.
  * Remove all opencv installations that were installed via **apt-get**.
  * Delete the __*.so__ files as well as the **build** dirctory within the **pupil_detectors** directory.
  * Start Pupil Capture. This should trigger a recompilation of the detector modules.


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

## 3D Eye Model Dependencies
```sh
sudo apt install -y libeigen3-dev
```

### Install Python Libraries

We recommend using a [virtual environment](https://docs.python.org/3/tutorial/venv.html) for running Pupil. To install all Python dependencies, you can use the [`requirements.txt`](https://github.com/pupil-labs/pupil/blob/master/requirements.txt) file from the root of the `pupil` repository.

```sh
# Upgrade pip to latest version. This is necessary for some dependencies.
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

**NOTE**: If you get the error `ImportError: No module named 'cv2'` when trying to run Pupil, please refer to the section [OpenCV Troubleshooting](#opencv-troubleshooting) above.

## Next Steps

That's it! You're done installing dependencies!

Now, you should be able to run Pupil from source. Follow the remaining instructions in the [README](../README.md).
