# Dependencies for macOS

These instructions have been tested for MacOS 10.8, 10.9, 10.10, 10.11, and 10.12.

## Install Apple Dev Tools

Trigger the install of the Command Line Tools (CLT) by typing this in your terminal and letting MacOS install the tools required:

```sh
git
```

## Install Homebrew
[Homebrew](http://brew.sh/) describes itself as "the missing package manager for OSX."  It makes development on MacOS much easier, [plus it's open source](https://github.com/Homebrew/homebrew).  Install with the following ruby script.

```sh
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

## Install Python 3.6 or higher

```sh
brew install python3
```

Add Homebrew installed executables and Python scripts to your path. Add the following two lines to your `~/.bash_profile`. (you can open textedit from the terminal like so: `open ~/.bash_profile`)

```sh
export PATH=/usr/local/bin:/usr/local/sbin:$PATH
export PYTHONPATH=/usr/local/lib/python3.6/site-packages:$PYTHONPATH
```

**Note:** You might need to change the Python path above depending on your installed version. `brew info python3 | grep site-packages` prints the corresponding site-packages folder.


## Other Dependencies with brew

Let's get started! Its time to put **brew** to work! Just copy paste commands into your terminal and listen to your machine purr.

```sh
brew install pkg-config
brew install libjpeg-turbo
brew install libusb
brew install portaudio
# opencv will install ffmpeg, numpy, and opencv-contributions automatically
# tbb is included by default with https://github.com/Homebrew/homebrew-core/pull/20101
brew install opencv
brew install glew
brew install glfw3
# dependencies for 2d_3d c++ detector
brew install ceres-solver
```

## libuvc
```
git clone --single-branch --branch build_fix_mac https://github.com/pupil-labs/libuvc
cd libuvc
mkdir build
cd build
cmake ..
make && make install
```

### Install Python Libraries

We recommend using a virtual environment with a valid installation of Python 3.6 or higher.

```sh
pip install cysignals
pip install cython
pip install msgpack==0.5.6
pip install numexpr
pip install packaging
pip install psutil
pip install pyaudio
pip install pyopengl
pip install pyzmq
pip install scipy
pip install torch torchvision
pip install git+https://github.com/zeromq/pyre

pip install pupil_apriltags
pip install git+https://github.com/pupil-labs/PyAV
pip install git+https://github.com/pupil-labs/pyuvc
pip install git+https://github.com/pupil-labs/pyndsi
pip install git+https://github.com/pupil-labs/pyglui
pip install git+https://github.com/pupil-labs/nslr
pip install git+https://github.com/pupil-labs/nslr-hmm
```

**NOTE:** Installing **pyglui** might fail on newer versions of **macOS** due to missing OpenGL headers. In this case, you need to install Xcode which comes with the required header files.

## Next Steps

That's it! You're done installing dependencies!

Now, you should be able to run Pupil from source. Follow the remaining instructions in the [README](../README.md).
