'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
video_capture is a module that extends opencv"s camera_capture for mac and windows
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
    from uvc_capture import Camera_Capture,device_list,CameraCaptureError
elif os_name == "Windows":
    from win_capture import Camera_Capture,device_list,CameraCaptureError
else:
    raise NotImplementedError()

from fake_capture import FakeCapture
from file_capture import File_Capture, FileCaptureError, EndofVideoFileError,FileSeekError


def autoCreateCapture(src,timestamps=None,timebase = None):
    preferred_idx = 0
    # checking src and handling all cases:
    src_type = type(src)

    if src_type is tuple:
        src,preferred_idx = src
        src_type = type(src)

    #looking for attached cameras that match the suggested names
    if src_type is list:
        matching_devices = []
        for device in device_list():
            if any([s in device['name'] for s in src]):
                matching_devices.append(device)

        if len(matching_devices) > preferred_idx:
            logger.info('Found %s as devices that match the src string pattern Using the %s match.'%([d['name'] for d in matching_devices],('first','second','third','fourth')[preferred_idx]) )
        else:
            if len(matching_devices) == 0:
                logger.error('No device found that matched %s'%src)
            else:
                logger.error('Not enough devices found that matched %s'%src)
            return FakeCapture(timebase=timebase)


        cap = Camera_Capture(matching_devices[preferred_idx]['uid'],timebase)
        logger.info("Camera selected: %s  with id: %s" %(matching_devices[preferred_idx]['name'],matching_devices[preferred_idx]['uid']))
        return cap

    #looking for attached cameras that match cv_id
    elif src_type is int:
        try:
            cap = device_list()[i]
        except IndexError, e:
            logger.warning('Camera with id %s not found.'%src_type)
            return FakeCapture(timebase=timebase)
        else:
            return Camera_Capture(cap['uid'],size,fps,timebase)


    #looking for videofiles
    elif src_type is str:
        if not isfile(src):
            logger.error('Could not locate VideoFile %s'%src)
            raise FileCaptureError('Could not locate VideoFile %s'%src)
        logger.info("Using %s as video source"%src)
        return File_Capture(src,timestamps=timestamps)
    else:
        logger.error("autoCreateCapture: Could not create capture, wrong src_type")
        return FakeCapture(size,fps,timebase=timebase)



