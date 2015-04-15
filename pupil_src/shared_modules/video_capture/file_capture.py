'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


import os,sys
import cv2
import numpy as np
from time import time,sleep
from pyglui import ui


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
        self.height,self.width,_ = img.shape
        self._gray = None

    def copy(self):
        return Frame(self.timestamp,self.img.copy(),self.index)

    @property
    def gray(self):
        if self._gray is None:
            self._gray =  cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        return self._gray

class File_Capture():
    """
    simple file capture.
    """
    def __init__(self,src,timestamps=None):
        self.menu = None
        self.auto_rewind = True
        self.controls = None #No UVC controls available with file capture
        self.sleep = 0.0
        # we initialize the actual capture based on cv2.VideoCapture
        self.cap = cv2.VideoCapture(src)
        self.src =src
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



    @property
    def frame_rate(self):
        #return rate as denominator only
        fps = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)
        if fps == 0:
            logger.error("Could not load media framerate info.")
        return fps


    @property
    def frame_size(self):
        width,height = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        if width == 0:
            logger.error("Could not load media size info.")
        return width,height


    def get_frame_index(self):
        return int(self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))

    def get_frame_count(self):
        if self.timestamps is None:
            logger.warning("No timestamps file loaded with this recording cannot get framecount")
            return None
        return len(self.timestamps)

    def get_frame(self):
        if self.sleep:
            sleep(self.sleep)
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
                logger.info("Filecapture is not a realtime source. -NOW- will be the current timestamp")
            except IndexError:
                logger.warning("timestamps not found.")
                timestamp = 0
        else:
            logger.info("Filecapture is not a realtime source. -NOW- will be the current time.")
            timestamp = time()
        return timestamp


    def init_gui(self,sidebar):
        self.menu = ui.Growing_Menu(label='File Capture Settings')
        self.menu.append(ui.Info_Text("Running Capture with '%s' as src"%self.src))
        self.menu.append(ui.Slider('sleep',self,min=0.,max=.2,label='add delay between frames (sec.)'))
        self.sidebar = sidebar
        self.sidebar.append(self.menu)

    def deinit_gui(self):
        if self.menu:
            self.sidebar.remove(self.menu)
            self.menu = None

    def close(self):
        self.deinit_gui()
