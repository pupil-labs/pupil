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
brew install cmake
brew install pkg-config
brew install libjpeg-turbo
brew install libusb
brew install portaudio
brew install ffmpeg
# opencv will install numpy, and opencv-contributions automatically
# tbb is included by default with https://github.com/Homebrew/homebrew-core/pull/20101
brew install opencv
brew install glew
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

We recommend using a [virtual environment](https://docs.python.org/3/tutorial/venv.html) for running Pupil. To install all Python dependencies, you can use the [`requirements.txt`](https://github.com/pupil-labs/pupil/blob/master/requirements.txt) file from the root of the `pupil` repository.

```sh
# Upgrade pip to latest version. This is necessary for some dependencies.
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

**NOTE:** Installing **pyglui** might fail on newer versions of **macOS** due to missing OpenGL headers. In this case, you need to install Xcode which comes with the required header files.

### OpenCV Troubleshooting
`ImportError: No module named 'cv2'`
  
When you see this error, Python cannot find the bindings from your OpenCV installation.

**We do NOT (!) recommend to install `opencv-python` via pip in that case!** 

Installing `opencv-python` will install another full (potentially different) version of opencv to your machine, so we are not recommending this setup.
When you install opencv with `brew install opencv` as we advise above, you should have the `cv2` package available for import in Python as this will install compatible Python bindings already.

However, you might run into these problems when using a virtual environment, as your virtual environment cannot by default access Python packages that were installed from `brew`.
In that case there are 2 options:

1. Symlink or copy the Python bindings into your virtualenv. See e.g. [step 4 of this stackoverflow post](https://stackoverflow.com/a/37190408) for reference.
2. Create your virtualenv with the [`--system-site-packages`](https://virtualenv.pypa.io/en/latest/userguide/#the-system-site-packages-option) option, which will enable access to system-installed Python packages.

## Next Steps

That's it! You're done installing dependencies!

Now, you should be able to run Pupil from source. Follow the remaining instructions in the [README](../README.md).
