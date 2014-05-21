'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
uvc_capture is a module that extends opencv"s camera_capture for mac and windows
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
if os_name == "Linux":
    from linux_video import Camera_Capture,Camera_List,CameraCaptureError
elif os_name == "Darwin":
    from mac_video import Camera_Capture,Camera_List,CameraCaptureError
else:
    from other_video import Camera_Capture,Camera_List,CameraCaptureError

from fake_capture import FakeCapture
from file_capture import File_Capture, FileCaptureError, EndofVideoFileError,FileSeekError


def autoCreateCapture(src,size=(640,480),fps=30,timestamps=None,timebase = None):
    # checking src and handling all cases:
    src_type = type(src)

    #looking for attached cameras that match the suggested names
    if src_type is list:
        matching_devices = []
        for device in Camera_List():
            if any([s in device.name for s in src]):
                matching_devices.append(device)

        if len(matching_devices) >1:
            logger.warning('Found %s as devices that match the src string pattern Using the first one.'%[d.name for d in matching_devices] )
        if len(matching_devices) ==0:
            logger.error('No device found that matched %s'%src)
            return FakeCapture(size,fps,timebase=timebase)


        cap = Camera_Capture(matching_devices[0],filter_sizes(matching_devices[0],size),fps,timebase)
        logger.info("Camera selected: %s  with id: %s" %(cap.name,cap.src_id))
        return cap

    #looking for attached cameras that match cv_id
    elif src_type is int:
        for device in Camera_List():
            if device.src_id == src:
                cap = Camera_Capture(device,filter_sizes(device,size),fps,timebase)
                logger.info("Camera selected: %s  with id: %s" %(cap.name,cap.src_id))
                return cap

        #control not supported for windows: init capture without uvc controls
        cap = Camera_Capture(src,size,fps,timebase)
        logger.warning('No UVC support: Using camera with id: %s'%src)
        return cap


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


def filter_sizes(cam,size):
    #here we can force some defaulit formats

    if "Integrated Camera" in cam.name:
        if size[0] == 640:
            logger.info("Lenovo Integrated camera selected. Forcing format to 640,480")
            return 640,480
        elif size[0] == 320:
            logger.info("Lenovo Integrated camera selected. Forcing format to 320,240")
            return 320,240

    return size


if __name__ == '__main__':
    cap = autoCreateCapture(1,(1280,720),30)
    if cap:
        print cap.controls
    print "done"