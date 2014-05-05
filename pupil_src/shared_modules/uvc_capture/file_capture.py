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
from time import time


#logging
import logging
logger = logging.getLogger(__name__)



class FileCaptureError(Exception):
    """General Exception for this module"""
    def __init__(self, arg):
        super(FileCaptureError, self).__init__()
        self.arg = arg

class EndofVideoFileError(Exception):
    """docstring for EndofVideoFileError"""
    def __init__(self, arg):
        super(EndofVideoFileError, self).__init__()
        self.arg = arg


class FileSeekError(Exception):
    """docstring for EndofVideoFileError"""
    def __init__(self):
        super(FileSeekError, self).__init__()


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

class File_Capture():
    """
    simple file capture.
    """
    def __init__(self,src,timestamps=None):
        self.auto_rewind = True
        self.controls = None #No UVC controls available with file capture
        # we initialize the actual capture based on cv2.VideoCapture
        self.cap = cv2.VideoCapture(src)
        if timestamps is None and src.endswith("eye.avi"):
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
            raise EndofVideoFileError("Reached end of video file.")
        if self.timestamps:
            try:
                timestamp = self.timestamps[idx]
            except IndexError:
                logger.warning("Reached end of timestamps list.")
                raise EndofVideoFileError("Reached end of timestamps list.")
        else:
            timestamp = time()
        return Frame(timestamp,img,index=idx)

    def seek_to_frame(self, seek_pos):
        if self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,seek_pos):
            offset = seek_pos - self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            if offset == 0:
                logger.debug("Seeked to frame: %s"%self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
                return
            # elif 0 < offset < 100:
            #     offset +=10
            #     if not self.seek_to_frame(seek_pos-offset):
            #         logger.warning('Could not seek to %s. Seeked to %s'%(seek_pos,self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)))
            #         return False
            #     logger.warning("Seek was not precice need to do manual seek for %s frames"%offset)
            #     while seek_pos != self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES):
            #         try:
            #             self.read()
            #         except EndofVideoFileError:
            #             logger.warning('Could not seek to %s. Seeked to %s'%(seek_pos,self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)))
            #     return True
            else:
                logger.warning('Could not seek to %s. Seeked to %s'%(seek_pos,self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)))
                raise FileSeekError()
        logger.error("Could not perform seek on cv2.VideoCapture. Command gave negative return.")
        raise FileSeekError()


    def get_now(self):
        idx = int(self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        if self.timestamps:

            try:
                timestamp = self.timestamps[idx]
                logger.warning("Filecapture is not a realtime source. -NOW- will be the current timestamp")
            except IndexError:
                logger.warning("timestamps not found.")
                timestamp = 0
        else:
            logger.warning("Filecapture is not a realtime source. -NOW- will be the current time.")
            timestamp = time()
        return timestamp

    def create_atb_bar(self,pos):
        return 0,0

    def kill_atb_bar(self):
        pass

    def close(self):
        pass
