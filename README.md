# Pupil
Open source eye tracking software platform that started as a thesis project at MIT. Pupil is a project in active, community driven development. Pupil mobile eye tracking hardware is accessible, hackable, and affordable. The software is open source and written in `Python` and `C++` when speed is an issue.

Our vision is to create tools for a diverse group of people interested in learning about eye tracking and conducting their eye tracking projects.

Chat with us on [Discord](https://pupil-labs.com/chat "#pupil channel on DiscordApp").

## Project Website
For an intro to the Pupil mobile eye tracking platform have a look at the [Pupil Labs Website](http://pupil-labs.com "Pupil Labs").

## Getting Started

<p align="center"><img src="https://via.placeholder.com/640x320?text=PLACEHOLDER"/></p>

## Building and Running Pupil from Source

If you want to develop a plugin or to extend Pupil for your project, this is the place to start.

If you have questions, encounter any problems, or want to share progress -- chat with us on the Pupil channel on [Discord](https://pupil-labs.com/chat). We will try our best to help you out, and answer questions quickly.

Pupil is a prototype and will continue to be in active development. If you plan to make changes to Pupil, want to see how it works, [fork the project on GitHub](https://github.com/pupil-labs/pupil/fork), install all dependencies and run Pupil source directly with Python.

### When is it recommended to run from source?
For a lot of applications it is sufficient to use our network api. In some cases it is justified to write custom plugins. Loading custom plugins during runtime is supported for the bundled applications as well. Be aware that the bundled applications only allow access to libraries that are already included in the bundle. Therefore, it is recommended to run from source if you develop a plugin or you make changes to the Pupil source code itself. This will also give you the advantage of receiving features and bug fixes as soon as they hit the Github repository.

### Installing Dependencies

The dependency setup for building and running Pupil from source is quite complex at the moment and very different for the different supported platforms, but we are actively working on making it easier to run from source. Please see the following documents on how to install and set up all necessary dependencies for your platform:

* [Ubuntu 18.04 LTS](./docs/dependencies-ubuntu18.md) (recommended Linux distribution)
* [Ubuntu 17.10 or lower](./docs/dependencies-ubuntu17.md)
* [MacOS](./docs/dependencies-macos.md)
* [Windows](./docs/dependencies-windows.md)

### Download and Run Pupil

After having set up all necessary dependencies, you can download and run Pupil.

For a quickstart you can just clone the github repository. If you created your own fork on please adjust the command appropriately.
```sh
git clone https://github.com/pupil-labs/pupil.git # or your fork
cd pupil
```

**Note:** If you are using Windows, you will have to do some more adjustments after downloading. Please see the [Dependencies for Windows](./docs/dependencies-windows.md) setup guide for a full workthrough.

Now you can fire up python to start Pupil Capture, Player or Service:

```sh
cd pupil_src
python main.py capture # or player/service
```




## License
All source code written by Pupil Labs is open for use in compliance with the [GNU Lesser General Public License (LGPL v3.0)](http://www.gnu.org/licenses/lgpl-3.0.en.html). We want you to change and improve the code -- make a fork! Make sure to share your work with the community! See the docs for more info on the [license](http://docs.pupil-labs.com/#license "License"). For support and custom licencing [contact us!](https://docs.pupil-labs.com/#email "email us")
