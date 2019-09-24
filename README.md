# Pupil
Open source eye tracking software platform that started as a thesis project at MIT. Pupil is a project in active, community driven development. Pupil mobile eye tracking hardware is accessible, hackable, and affordable. The software is open source and written in `Python` and `C++` when speed is an issue.

Our vision is to create tools for a diverse group of people interested in learning about eye tracking and conducting their eye tracking projects.

Chat with us on [Discord](https://pupil-labs.com/chat "#pupil channel on DiscordApp").

## Project Website
For an intro to the Pupil mobile eye tracking platform have a look at the [Pupil Labs Website](http://pupil-labs.com "Pupil Labs").

## Getting Started

<p align="center"><img src="https://via.placeholder.com/640x320?text=PLACEHOLDER"/></p>

## Building and Running Pupil from Source

### Installing Native Dependencies

The dependency setup for building and running Pupil from source is quite complex at the moment, but we are actively working on making it easier to run from source. Please see the following documents on how to install and set up all necessary dependencies for your platform:

* [Ubuntu 18.04 LTS](./docs/dependencies-ubuntu18.md) (recommended Linux distribution)
* [Ubuntu 17.10 or lower](./docs/dependencies-ubuntu17.md)
* [MacOS](./docs/dependencies-macos.md)
* [Windows](./docs/dependencies-windows.md)


### Installing Python Libraries

We recommend using a virtual environment with a valid installation of Python 3.6 or higher.

```sh
pip install cysignals
pip install cython
pip install msgpack==0.5.6
pip install numexpr
pip install psutil
pip install pyaudio
pip install pyopengl
pip install pyzmq
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



## License
All source code written by Pupil Labs is open for use in compliance with the [GNU Lesser General Public License (LGPL v3.0)](http://www.gnu.org/licenses/lgpl-3.0.en.html). We want you to change and improve the code -- make a fork! Make sure to share your work with the community! See the docs for more info on the [license](http://docs.pupil-labs.com/#license "License"). For support and custom licencing [contact us!](https://docs.pupil-labs.com/#email "email us")
