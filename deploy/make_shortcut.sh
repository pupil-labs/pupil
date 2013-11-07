#!/bin/sh

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $BASEDIR

TEXT="
[Desktop Entry]\n
Version=1.0\n
Name=Pupil Capture\n
Comment=Pupil Capture Software\n
Exec= '${BASEDIR}/pupil_capture/pupil_capture'\n
Icon= ${BASEDIR}/pupil_capture/icon.ico\n
Terminal=true\n
Type=Application\n
Categories=Application;"


echo $TEXT > pupil_capture.desktop
chmod 775 pupil_capture.desktop