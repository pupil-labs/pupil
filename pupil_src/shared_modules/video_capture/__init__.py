'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
video_capture is a module that extends opencv's camera_capture for mac and windows
on Linux it repleaces it completelty.
it adds some fuctionalty like:
    - access to all uvc controls
    - assosication by name patterns instead of id's (0,1,2..)
it requires:
    - opencv 2.3+
    - on Linux: v4l2-ctl (via apt-get install v4l2-util)
    - on MacOS: uvcc (binary is distributed with this module)
"""
import os,sys
import cv2
import numpy as np
from os.path import isfile
from time import time

import platform
os_name = platform.system()
del platform

#logging
import logging
logger = logging.getLogger(__name__)


###OS specific imports and defs
if os_name in ("Linux","Darwin"):
    from uvc_capture import Camera_Capture,device_list,CameraCaptureError,is_accessible
elif os_name == "Windows":
    from win_video import Camera_Capture,device_list,CameraCaptureError
    def is_accessible(uid):
        return True
else:
    raise NotImplementedError()

from av_file_capture import File_Capture, FileCaptureError, EndofVideoFileError,FileSeekError


def autoCreateCapture(src,timestamps=None,timebase = None):
    '''
    src can be one of the following:
     - a path to video file
     - patter of name matches
     - a device index
     - None
    '''
    # video source
    if type(src) is str and os.path.isfile(src):
        return File_Capture(src,timestamps=timestamps)

    # live src - select form idx
    if type(src) == int:
        try:
            uid = device_list()[src]['uid']
        except IndexError:
            logger.warning("UVC Camera at index:'%s' not found."%src)
            src = None
        else:
            if is_accessible(uid):
                logger.info("UVC Camera with id:'%s' selected."%src)
                return Camera_Capture(uid,timebase=timebase)
            else:
                logger.warning("Camera selected by id matches is found but already in use")
                src = None

    # live src - select form pattern
    elif type(src) in (list,tuple):
        src = uid_from_name(src)

    # fake capture
    if src is None:
        logger.warning("Starting with Fake_Capture.")

    return Camera_Capture(src,timebase=timebase)



def uid_from_name(pattern):
    # looking for attached cameras that match the suggested names
    # give precedence to camera that matches the first pattern in list.
    matching_devices = []
    attached_devices = device_list()
    for name_pattern in pattern:
        for device in attached_devices:
            if name_pattern in device['name']:
                if is_accessible(device['uid']):
                    return device['uid']
                else:
                    logger.warning("Camera '%s' matches the pattern but is already in use"%device['name'])
    logger.error('No accessible device found that matched %s'%pattern)


