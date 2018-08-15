import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pupil",
    version="2.0a1",
    author="pupil labs",
    author_email="info@pupil-labs.com",
    description="Open source eye tracking ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pupil-labs/pupil",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "pupil_capture=pupil.apps.capture:main",
            "pupil_service=pupil.apps.service:main",
            "pupil_player=pupil.apps.player:main",
        ]
    },
    classifiers=(
        "Development Status :: 1 - Planning",
        "Environment :: X11 Applications",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ),
)
