# Dependencies for Windows

## System Requirements

We develop the Windows version of Pupil using **64 bit** **Windows 10**.

Therefore we can only debug and support issues for **Windows 10**.

## Notes Before Starting

### 64 Bit
You should be using a 64 bit system and therefore all downloads, builds, and libraries should be for `x64` unless otherwise specified.

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

## 7-Zip
Install [7-zip](http://www.7-zip.org/download.html) to extract files.

## Git
Install [Git](https://git-scm.com/download/win) to clone the Pupil source code repository in the end.

**NOTE:** If you run Pupil from source, it needs to be downloaded via git. Downloading only the source code won't work, as Pupil infers its version from the last git tag when not running from bundle.

## Python

You will need a **64 bit version** of Python 3.6, e.g. download [Python 3.6.8](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe). If you install any other version, make sure to install the **64 bit version!**

**NOTE:** Currently our build process for WIndows does not yet support Python 3.7 as pyaudio does not yet have prebuild wheels for Windows available for Python 3.7. You are thus highly encouraged to use the latest stable version of Python 3.6.

If you downloaded the linked installer:

- Run the Python installer.
- Check the box `Add Python to PATH`. This will add Python to your System PATH Environment Variable.
- Check the box `Install for all users`. **Note:** By default this will install Python to `C:\Program Files\Python36`. Some build scripts may fail to start Python due to spaces in the path name. So, you may want to consider installing Python to `C:\Python36` instead.

## Install Python Libraries

We recommend using a [virtual environment](https://docs.python.org/3/tutorial/venv.html) for running Pupil. To install all Python dependencies, you can use the [`requirements.txt`](https://github.com/pupil-labs/pupil/blob/master/requirements.txt) file from the root of the `pupil` repository.

```sh
# Upgrade pip to latest version. This is necessary for some dependencies.
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

**NOTE:** `pyuvc` requires that you download Microsoft Visual C++ 2010 Redistributable from [microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=14632). The `pthreadVC2` lib, which is used by libuvc, depends on `msvcr100.dll`.

## Modifying Pupil to Work with Windows

Before you can start using Pupil from source on Windows, you will have to make a few additional changes to the repository.

### Clone Pupil Repository

Open a command prompt where you want to clone the git repository, e.g. `C:\work\pupil`
Make sure you don't have spaces in your path to the repository as this has been repeatedly causing issues for users.
Then run:
```sh
git clone https://github.com/pupil-labs/pupil.git
```

### Include pupil_external in PATH Variable

- Follow the instructions under the System Environment Variables section above to add a new environment variable to PATH
- Add the full path to the `pupil_external` folder of the repository that you just cloned, e.g. `C:\work\pupil\pupil_external`
- You might have to restart your computer so that the PATH variable is refreshed

### Setup pupil_external Dependencies
The following steps require you to store dynamic libraries in the `pupil_external` folder of the cloned repository so that you do not have to add further modifications to your system PATH.

#### GLEW

- Download GLEW Windows binaries from [sourceforge](http://glew.sourceforge.net/)
- Unzip GLEW in your work dir
- Copy `glew32.dll` to `pupil_external`

#### FFMPEG

- [Download FFMPEG v4.3 Windows **shared** binaries](https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2020-12-08-13-03/ffmpeg-n4.3.1-26-gca55240b8c-win64-lgpl-shared-4.3.zip)
- Unzip `ffmpeg-*-shared-4.3.zip` to your work dir
- Copy the following 8 `.dll` files to `pupil_external`
    - `avcodec-58.dll`
    - `avdevice-58.dll`
    - `avfilter-7.dll`
    - `avformat-58.dll`
    - `avutil-56.dll`
    - `postproc-55.dll`
    - `swresample-3.dll`
    - `swscale-5.dll`

## Start Pupil

To start either of the applications -- Capture, Player, or Service -- you need to run
the `main.py` file with the respective application name as argument.

```sh
cd pupil_src
python main.py capture # or player/service
```
