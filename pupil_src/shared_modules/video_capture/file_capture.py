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
    def __init__(self, timestamp,img,index):
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
        self.freerun = False
        self.timestamps = None
        self.display_time = 0

        # we initialize the actual capture based on cv2.VideoCapture
        self.cap = cv2.VideoCapture(src)
        self.src = src

        #load/generate timestamps.
        if timestamps is None:
            timestamps_path,ext =  os.path.splitext(src)
            timestamps = timestamps_path+'_timestamps.npy'
        try:
            self.timestamps = np.load(timestamps).tolist()
        except IOError:
            logger.warning("did not find timestamps file, making timetamps up based on fps and frame count.")
            frame_rate = float(self.frame_rate)
            if frame_rate == 0:
                logger.warning("Framerate not available - setting to 30fps.")
                frame_rate = 30.0
            self.timestamps = [i/frame_rate for i in xrange(self.get_frame_count())]
        else:
            logger.debug("loaded %s timestamps from %s"%(len(self.timestamps),timestamps))


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
            frame_count = int(self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            if frame_count:
                return frame_count
            else:
                logger.warning("Cannot get framecount")
                return None
        return len(self.timestamps)

    def get_frame_nowait(self):
        idx = int(self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        s, img = self.cap.read()
        if not s:
            logger.warning("Reached end of video file.")
            raise EndofVideoFileError("Reached end of video file.")
        try:
            timestamp = self.timestamps[idx]
        except IndexError:
            logger.warning("Reached end of timestamps list.")
            raise EndofVideoFileError("Reached end of timestamps list.")
        self.show_time = timestamp
        return Frame(timestamp,img,idx)

    def wait(self,frame):
        if self.display_time:
            wait_time  = frame.timestamp - self.display_time - time()
            if 1 > wait_time > 0 :
                sleep(wait_time)
        self.display_time = frame.timestamp - time()


    def get_frame(self):
        frame = self.get_frame_nowait()
        self.wait(frame)
        return frame


    def seek_to_frame(self, seek_pos):
        if self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,seek_pos):
            offset = seek_pos - self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            if offset == 0:
                logger.debug("Seeked to frame: %s"%self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
                self.display_time = 0
                return
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

    def get_timestamp():
        return self.get_now()



    def init_gui(self,sidebar):
        self.menu = ui.Growing_Menu(label='File Capture Settings')
        self.menu.append(ui.Info_Text("Running Capture with '%s' as src"%self.src))
        self.sidebar = sidebar
        self.sidebar.append(self.menu)

    def deinit_gui(self):
        if self.menu:
            self.sidebar.remove(self.menu)
            self.menu = None

    def close(self):
        self.deinit_gui()
