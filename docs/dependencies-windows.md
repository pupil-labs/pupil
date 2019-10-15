# Dependencies for Windows

## System Requirements

We develop the Windows version of Pupil using **64 bit** **Windows 10**.

Therefore we can only debug and support issues for **Windows 10**.

## Notes Before Starting

### Work Directory - Very Important!
  
We will make a directory called `work` at `C:\work` and will use this directory for all build processes and setup scripts. Whenever we refer to the `work` directory, it will refer to `C:\work`. You can change this to whatever is convenient for you, but note that many setup files have hardcoded references to `C:\work`, so changing this won't run without a lot of adjustments.

If for whatever reason you still don't want to use `C:\work`, you might want to consider creating a symbolic link at `C:\work` to your actual development folder. This might save you some work.

### Command Prompt
  
We will **always** be using `x64 Native Tools Command Prompt for VS 2017` as our command prompt. Make sure to only use this command prompt. Unlike unix systems, windows has many possible "terminals" or "cmd prompts". We are targeting `x64` systems and require the `x64` command prompt. You can access this cmd prompt from the Visual Studio 2017 shortcut in your Start menu.

### 64 Bit
You should be using a 64 bit system and therefore all downloads, builds, and libraries should be for `x64` unless otherwise specified.

### Windows Paths and Python

Windows uses backslashes `\` as path separators. In Python, this is a special "escape" character. When specifying Windows paths in a Python string you must either use `\\` instead of `\` or use [Python raw strings](https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals), e.g. `r"\"`.

### System Environment Variables

You will need to check to see that Python was added to your system PATH variables. You will also need to manually add other entries to the system PATH later in the setup process.

To access your System Environment Variables:

- Right click on the Windows icon in the system tray.
- Select `System`.
- Click on `Advanced system settings`.
- Click on `Environment Variables...`.
- You can click on `Path` in `System Variables` to view the variables that have been set.
- You can `Edit` or `Add` new paths (this is needed later in the setup process).

### Help

For discussion or questions on Windows head over to our [#pupil Discord channel](https://discord.gg/gKmmGqy). If you run into trouble please raise an [issue on github](https://github.com/pupil-labs/pupil)!

## Install Visual Studio

Download Visual Studio 2017 Community version 15.8 from [visualstudio.com](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community)

- Run the Visual Studio bootstrapper `.exe`.
- Navigate to the `Workloads` tab
- In the `Workloads` tab, choose `Desktop Development with C++`. This will install all runtimes and components we need for development. Here is a list of what you should see `checked` in the `Desktop development with C++` in the `Summary` view:
    - VC++ 2017 v141 toolset (x86,x64)
    - C++ profiling tools
    - Windows 10 SDK (10.0.15063.0) for Desktop C++ x86 and x64
    - Visual C++ tools for CMAKE
    - Visual C++ ATL support
    - MFC and ATL support (x86, x64)
    - Standard Library Modules
    - VC++ 2015.3 v140 toolset for desktop (x86, x64)
- Navigate to the `Individual Components` tab
- In the `Individual Components` tab check `Git`. This will install `git` on your system. In the Summary Panel for `Individual Components` you should see:
  - `Git for Windows`
- Click `Install`

**Note:** You can always re-run the Visual Studio bootstrapper to modify your Visual Studio installation.

## 7-Zip
Install [7-zip](http://www.7-zip.org/download.html) to extract files.

## Python

You will need a **64 bit version** of Python 3.6 or above, e.g. download [Python 3.6.8](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe). If you install any other version, make sure to install the **64 bit version!**

**NOTE:** Currently our build process for WIndows does not yet support Python 3.7 as pyaudio does not yet have prebuild wheels for Windows available for Python 3.7. You are thus highly encouraged to use the latest stable version of Python 3.6.

If you downloaded to linked installer:

- Run the Python installer.
- Check the box `Add Python to PATH`. This will add Python to your System PATH Environment Variable.
- Check the box `Install for all users`. **Note:** By default this will install Python to `C:\Program Files\Python36`. Some build scripts may fail to start Python due to spaces in the path name. So, you may want to consider installing Python to `C:\Python36` instead.


## Install Python Libraries

```sh
pip install cython
pip install msgpack==0.5.6
pip install numexpr
pip install opencv-python==3.*
pip install packaging
pip install psutil
pip install pyaudio
pip install pyopengl
pip install pyzmq
pip install scipy
pip install win_inet_pton
pip install git+https://github.com/zeromq/pyre

pip install pupil_apriltags
pip install git+https://github.com/pupil-labs/nslr
pip install git+https://github.com/pupil-labs/nslr-hmm
```

Now install pytorch:
- Open the pytorch website for local installation: https://pytorch.org/get-started/locally/
- Select options: Stable, Windows, Pip, Python 3.6, CUDA 9.0.
- You will be provided with two commands. Run them in the order given to install the wheels.

## Pupil Labs Python Wheels

In addition to these libraries, you will need to install some Pupil-Labs support libraries. Since building them for Windows is also not automated yet, we provide some prebuilt wheels that you can use. If you want to build the support libraries yourself as well, you will have to look for install instructions on the respective GitHub repositories.

Download the following Python wheels from Pupil Labs github repos:

- [pyglui](https://github.com/pupil-labs/pyglui/releases/latest)
- [pyav](https://github.com/pupil-labs/pyav/releases/latest)
- [pyndsi](https://github.com/pupil-labs/pyndsi/releases/latest)
- [pyuvc](https://github.com/pupil-labs/pyuvc/releases/latest)

`pyuvc` requires that you download Microsoft Visual C++ 2010 Redistributable from [microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=14632). The `pthreadVC2` lib, which is used by libuvc, depends on `msvcr100.dll`.

Open your command prompt and `Run as administrator` in the directory where the wheels are downloaded.

- Install all wheels with `pip install X` (where X is the name of the `.whl` file)
- You can check that libs are installed with `python import X` statements in the command prompt where `X` is the name of the lib.

## Ceres for Windows
Navigate to your work directory

- `git clone --recursive https://github.com/tbennun/ceres-windows.git`
- Download [Eigen 3.3.3](https://bitbucket.org/eigen/eigen/downloads/?tab=tags)
- Unzip Eigen and rename the extracted `eigen-eigen-67e894c6cd8f` directory to `Eigen`
- Copy the `Eigen` directory into `ceres-windows`
- Copy `C:\work\ceres-windows\ceres-solver\config\ceres\internal\config.h` to `C:\work\ceres-windows\ceres-solver\include\ceres\internal`
- Open `ceres-2015.sln` and with Visual Studio 2017 and agree to update to 2017.
- Set configurations to `Release` and `x64`
- Right click on `libglog_static` and `Build`
- Right click on `ceres_static` and `Build`

## Modifying Pupil to Work with Windows

Before you can start using Pupil from source on Windows, you will have to make a few additional changes to the repository.

### Clone Pupil Repository

Open a command prompt in your work dir and:
```sh
git clone https://github.com/pupil-labs/pupil.git
```

### Include pupil_external in PATH Variable

- Follow the instructions under the System Environment Variables section above to add a new environment variable to PATH
- Add the following folder: `C:\work\pupil\pupil_external`
- You might have to restart your computer so that the PATH variable is refreshed

### Setup pupil_external Dependencies
The following steps require you to store dynamic libraries in the `pupil_external` folder of the cloned repository so that you do not have to add further modifications to your system PATH.

#### GLEW

- Download GLEW Windows binaries from [sourceforge](http://glew.sourceforge.net/)
- Unzip GLEW in your work dir
- Copy `glew32.dll` to `pupil_external`

#### GLFW

- Download GLFW Windows binaries from [glfw.org](http://www.glfw.org/download.html)
- Unzip GLFW to your work dir
- Copy `glfw3.dll` from `lib-vc2015` to `pupil_external`

#### FFMPEG

- Download FFMPEG v4.0 Windows shared binaries from [ffmpeg](http://ffmpeg.zeranoe.com/builds/)
- Unzip ffmpeg-shared to your work dir
- Copy the following 8 `.dll` files to `pupil_external`
    - `avcodec-58.dll`
    - `avdevice-58.dll`
    - `avfilter-7.dll`
    - `avformat-58.dll`
    - `avutil-56.dll`
    - `postproc-55.dll`
    - `swresample-3.dll`
    - `swscale-5.dll`

#### OpenCV

- Download opencv 3.4.5 exe installer from [sourceforge](https://sourceforge.net/projects/opencvlibrary/files/3.4.5/opencv-3.4.5-vc14_vc15.exe/download)
- Unzip OpenCV to your work dir and rename dir to `opencv`
- Copy `opencv\build\x64\vc14\bin\opencv_world345.dll` to `pupil_external`

### Modify pupil_detectors setup.py

- Open `pupil\pupil_src\shared_modules\pupil_detectors\setup.py`
- Go to the `if platform.system() == 'Windows'` block
- Check that paths for `opencv`, `Eigen` and `ceres-windows` are correctly specified. The installed opencv lib is `opencv_world345.lib`.
- Edit paths if necessary
- Save and close setup.py

**Note:** When starting run_capture.bat, it will build the pupil_detectors module. However, if you are debugging, you may want to try building explicitly. From within `pupil/pupil_src/shared_modules/pupil_detectors` run `python setup.py build` to build the pupil_detectors.

In case you are using Visual Studio 2017 with v15.8 or v15.9 update, you may encounter an error regarding _ENABLE_EXTENDED_ALIGNED_STORAGE while building. Please refer to the fix [here](https://github.com/pupil-labs/pupil/issues/1331#issuecomment-430418074).

### Modify optimization_calibration setup.py

- Open `pupil\pupil_src\shared_modules\calibration_routines\optimization_calibration\setup.py`
- Go to the `if platform.system() == 'Windows'` block
- Check that paths for `opencv`, `Eigen` and `ceres-windows` are correctly specified. The installed opencv lib is `opencv_world345.lib`.
- Edit paths if necessary
- Save and close setup.py

**Note:** When starting run_capture.bat, it will build the optimization_calibration module. However, if you are debugging, you may want to try building explicitly. From within `pupil/pupil_src/shared_modules/calibration_routines/optimization_calibration` run `python setup.py build` to build the optimization_calibration module.

## Start Pupil

To start either of the applications -- Capture, Player, or Service -- you can to execute the respective `run_*.bat` file (e.g. by double clicking), i.e. `run_capture.bat`, `run_player.bat`, or `run_service.bat`.

```powershell
cd pupil_src
run_capture.bat
# or run_player.bat
# or run_service.bat
```

You can also run `main.py` directly from your IDE or command line. Make sure that the `pupil_external` folder is part of your system path.
```sh
cd pupil_src
python main.py capture # or player/service
```
