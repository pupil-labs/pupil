# Pupil
<p align="center"><img src="https://via.placeholder.com/800x400?text=PLACEHOLDER"/></p>

**Open source eye tracking platform.**

Pupil is a project in active, community driven development. Pupil Core mobile eye tracking hardware is accessible, hackable, and affordable. The software is open source and written in `Python` and `C++` when speed is an issue.


Our vision is to create tools for a diverse group of people interested in learning about eye tracking and conducting their eye tracking projects.

Chat with us on [Discord](https://pupil-labs.com/chat "Pupil Server on Discord").

## Website
For an intro to Pupil Core mobile eye tracking platform take a look at the [Pupil Labs Website](http://pupil-labs.com/products/core "Pupil Labs").

## Documentation
Read the user guide and developer docs at [docs.pupil-labs.com](https://docs.pupil-labs.com "Pupil Labs user guide and develper docs"). 

## Users
[Download the latest apps](https://github.com/pupil-labs/pupil/releases/latest "Download Pupil Capture, Pupil Player, and Pupil Service application bundles")! You don't need to know how to write code to _use_ Pupil.

## Develelopers
There are a number of ways you can extend the functionality of Pupil.
- [Use the API](): Use the network based real-time API to communicate with Pupil over the network and integrate with your application. 
- [Develop a Plugin](): Plugins are loaded at runtime from the app bundle. You don't need to setup dependencies and run from source unless you really need bare metal access.
- [Run from Source](#installing-dependencies): Can't do what you need to do with the network based api or plugin? Then get ready to dive into the inner workings of Pupil and run from source!

### Installing Dependencies
* [Ubuntu 18.04 LTS](./docs/dependencies-ubuntu18.md "Pupil dependency installation for Ubuntu 18.04") (recommended Linux distribution)
* [Ubuntu 17.10 or lower](./docs/dependencies-ubuntu17.md "Pupil dependency installation for Ubuntu 17.10 or lower")
* [macOS](./docs/dependencies-macos.md "Pupil dependency installation for macOS")
* [Windows 10](./docs/dependencies-windows.md "Pupil dependency installation for Windows 10")

### Running from source
After you have installed all dependencies, clone this repo and start Pupil software.

**macOS and Linux**

```sh
# Pupil Capture
python3 pupil_src/main.py

# Pupil Player
python3 pupil_src/main.py player

# Pupil Service
python3 pupil_src/main.py service
```

**Windows**
Use the `.bat` files to run Capture, Player, and Service.

## License
All source code written by Pupil Labs is open for use in compliance with the [GNU Lesser General Public License (LGPL v3.0)](http://www.gnu.org/licenses/lgpl-3.0.en.html). We want you to change and improve the code -- make a fork! Make sure to share your work with the community! See the docs for more info on the [license](http://docs.pupil-labs.com/#license "License"). For support and custom licencing [contact us!](https://docs.pupil-labs.com/#email "email us")
