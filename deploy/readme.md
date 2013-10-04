
workflow:

bundle using pyinstaller:
pyinstaller -w bundle_name.spec

create a version file inside the dicrtiubution folder:
python write_version_string.py

make sure that all excecutables in /dist/pupil are chmodded to be exceutable:
chmod 775 pupil_capture
chmod 775 v4l2-ctl



