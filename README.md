# Pupil
<p align="center"><img src="https://raw.githubusercontent.com/wiki/pupil-labs/pupil/media/images/pupil_labs_pupil_core_repo_banner.jpg"/></p>

**Open source eye tracking platform.**

Pupil is a project in active, community driven development. Pupil Core mobile eye tracking hardware is accessible, hackable, and affordable. The software is open source and written in `Python` and `C++` when speed is an issue.


Our vision is to create tools for a diverse group of people interested in learning about eye tracking and conducting their eye tracking projects.

Chat with us on [Discord](https://pupil-labs.com/chat "Pupil Server on Discord").

## Website
For an intro to Pupil Core mobile eye tracking platform take a look at the [Pupil Labs Website](http://pupil-labs.com/products/core "Pupil Labs").

## Users
[Download the latest apps](https://github.com/pupil-labs/pupil/releases/latest "Download Pupil Capture, Pupil Player, and Pupil Service application bundles")! 

You don't need to know how to write code to _use_ Pupil. Read the [Pupil Core user guide](https://docs.pupil-labs.com/core/ "Pupil Core user guide"). 

## Developers
There are a number of ways you can extend the functionality of Pupil.
- [Use the API](https://docs.pupil-labs.com/developer/core/network-api/): Use the network based real-time API to communicate with Pupil over the network and integrate with your application. 
- [Develop a Plugin](https://docs.pupil-labs.com/developer/core/plugin-api/): Plugins are loaded at runtime from the app bundle. You don't need to setup dependencies and run from source unless you need libraries that are not included in the app bundle. 
- [Run from Source](#installing-dependencies): Can't do what you need to do with the network based api or plugin? Then get ready to dive into the inner workings of Pupil and run from source!

All setup and dependency installation instructions are contained in this repo. All other developer documentation is [here](https://docs.pupil-labs.com/developer/core "Pupil Core developer docs").

### Installing Dependencies
- [Ubuntu 18.04 LTS](./docs/dependencies-ubuntu18.md "Pupil dependency installation for Ubuntu 18.04") (recommended Linux distribution)
- [Ubuntu 17.10 or lower](./docs/dependencies-ubuntu17.md "Pupil dependency installation for Ubuntu 17.10 or lower")
- [macOS](./docs/dependencies-macos.md "Pupil dependency installation for macOS")
- [Windows 10](./docs/dependencies-windows.md "Pupil dependency installation for Windows 10")

### Clone the repo
After you have installed all dependencies, clone this repo and start Pupil software.

```sh
git clone https://github.com/pupil-labs/pupil.git # or your fork
cd pupil
```

_Note_: If you are using Windows, you will have to complete a few more steps after cloning the repo. Please refer to the [Windows 10 dependencies setup guide](./docs/dependencies-windows.md "Pupil dependency installation for Windows 10").

### Run Pupil
Open your terminal and use Python to start Pupil Capture, Player, or Service.

```sh
cd pupil_src
python main.py capture # or player/service
```

## License
All source code written by Pupil Labs is open for use in compliance with the [GNU Lesser General Public License (LGPL v3.0)](http://www.gnu.org/licenses/lgpl-3.0.en.html). We want you to change and improve the code -- make a fork! Make sure to share your work with the community!
