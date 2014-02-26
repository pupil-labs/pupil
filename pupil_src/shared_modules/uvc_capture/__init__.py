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
    from linux_video import Camera_Capture,Camera_List
elif os_name == "Darwin":
    from mac_video import Camera_Capture,Camera_List
else:
    from other_video import Camera_Capture,Camera_List


# non os specific defines
class Frame(object):
    """docstring of Frame"""
    def __init__(self, timestamp,img,index=None,compressed_img=None, compressed_pix_fmt=None):
        self.timestamp = timestamp
        self.index = index
        self.img = img
        self.compressed_img = compressed_img
        self.compressed_pix_fmt = compressed_pix_fmt

    def copy(self):
        return Frame(self.timestamp,self.img.copy(),self.index)

class FileCapture():
    """
    simple file capture.
    """
    def __init__(self,src,timestamps=None):
        self.auto_rewind = True
        self.controls = None #No UVC controls available with file capture
        # we initialize the actual capture based on cv2.VideoCapture
        self.cap = cv2.VideoCapture(src)
        if timestamps is None:
            timestamps_loc = os.path.join(src.rsplit(os.path.sep,1)[0],'eye_timestamps.npy')
            logger.debug("trying to auto load eye_video timestamps with video at: %s"%timestamps_loc)
        else:
            timestamps_loc = timestamps
            logger.debug("trying to load supplied timestamps with video at: %s"%timestamps_loc)
        try:
            self.timestamps = np.load(timestamps_loc).tolist()
            logger.debug("loaded %s timestamps"%len(self.timestamps))
        except:
            logger.debug("did not find timestamps")
            self.timestamps = None


    def get_size(self):
        width,height = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        if width == 0:
            logger.error("Could not load media size info.")
        return width,height

    def set_fps(self):
        logger.warning("You cannot set the Framerate on this File Capture")

    def get_fps(self):
        fps = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)
        if fps == 0:
            logger.error("Could not load media framerate info.")
        return fps

    def get_frame_index(self):
        return int(self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))

    def get_frame_count(self):
        if self.timestamps is None:
            logger.warning("No timestamps file loaded with this recording cannot get framecount")
            return None
        return len(self.timestamps)

    def get_frame(self):
        idx = int(self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        s, img = self.cap.read()
        if not s:
            logger.warning("Reached end of video file.")
            return Frame(None,None)
        if self.timestamps:
            try:
                timestamp = self.timestamps[idx]
            except IndexError:
                logger.warning("Reached end of timestamps list.")
                return Frame(None,None)
        else:
            timestamp = time()
        return Frame(timestamp,img,index=idx)

    def seek_to_frame(self, seek_pos):
        logger.debug("seeking to frame: %s"%seek_pos)
        if self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,seek_pos):
            return True
        logger.error("Could not perform seek on cv2.VideoCapture. Command gave negative return.")
        return False


    def seek_to_frame_prefetch(self, seek_pos):
        prefetch = 10
        logger.debug("seeking to frame: %s"%seek_pos)
        if self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,int(seek_pos)-prefetch):
            while self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) < seek_pos:
                print "seek:",self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                s,_=self.cap.read()
                if not s:
                    logger.error("Could not seek to position %s" %seek_pos)
                    return
                prefetch -=1
                if prefetch < -10:
                    logger.error("Could not seek to position %s stepped out of prefetch" %seek_pos)
                    return
            logger.debug("Sucsessful seek to %s" %self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
            return True
        logger.error("Could not perform seek on cv2.VideoCapture. Command gave negative return.")
        return

    def create_atb_bar(self,pos):
        return 0,0

    def kill_atb_bar(self):
        pass

    def close(self):
        pass


def autoCreateCapture(src,size=(640,480),fps=30,timestamps=None):
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
            return

        cap = Camera_Capture(matching_devices[0],filter_sizes(matching_devices[0],size),fps)
        logger.info("Camera selected: %s  with id: %s" %(cap.name,cap.src_id))
        return cap

    #looking for attached cameras that match cv_id
    elif src_type is int:
        for device in Camera_List():
            if device.src_id == src:
                cap = Camera_Capture(device,filter_sizes(device,size),fps)
                logger.info("Camera selected: %s  with id: %s" %(cap.name,cap.src_id))
                return cap

        #control not supported for windows: init capture without uvc controls
        cap = Camera_Capture(src,size,fps)
        logger.warning('No UVC support: Using camera with id: %s'%src)
        return cap


    #looking for videofiles
    elif src_type is str:
        if not isfile(src):
            logger.error('Could not locate VideoFile %s'%src)
            return
        logger.info("Using %s as video source"%src)
        return FileCapture(src,timestamps=timestamps)
    else:
        raise Exception("autoCreateCapture: Could not create capture, wrong src_type")


def filter_sizes(cam,size):
    #here we can force some defaulit formats

    if "Integrated Camera" in cam.name:
        if size[0] == 640:
            logger.info("Lenovo Integrated camera selected. Forceing format to 640,480")
            return 640,480
        elif size[0] == 320:
            logger.info("Lenovo Integrated camera selected. Forceing format to 320,240")
            return 320,240
    else:
        return size


if __name__ == '__main__':
    cap = autoCreateCapture(1,(1280,720),30)
    if cap:
        print cap.controls
    print "done"