'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import videoInput as vi
assert vi.VERSION >= 0.1
import numpy as np
import math
from time import time
import logging
from videoInput import CaptureSettings, DeviceSettings
from sqlite3 import Timestamp
logger = logging.getLogger(__name__)

class CameraCaptureError(Exception):
    """General Exception for this module"""
    def __init__(self, arg):
        super(CameraCaptureError, self).__init__()
        self.arg = arg
        
def Camera_List():
    '''
    Thin wrapper around list_devices to improve formatting.
    '''
    class Cam(object):
        def __init__(self, d):
            self.device = d
        
        @property
        def name(self):
            return self.device.friendlyName
        
        @property
        def src_id(self):
            return self.device.symbolicName
    
    devices = vi.DeviceList()
    _getVideoInputInstance().getListOfDevices(devices)
    
    cam_list = []
    for d in devices:
        cam_list.append(Cam(d))
    return cam_list

class Frame(object):
    timestamp = 0
    width = 0
    height = 0
    
    _npy_frame = None
    
    def __init__(self, npy_frame):
        self._npy_frame = npy_frame
        self.height, self.width, _ = npy_frame.shape
    
    @property
    def img(self):
        return self._npy_frame
    
    @property
    def gray(self):
        return None
    
    @property
    def bgr(self):
        return self._npy_frame

class Camera_Capture(object):
    """
    Camera Capture encapsulates the videoInput class
    """ 
    deviceSettings = None
    captureSettings = None
    device = None
    stream = None
    
    width = 640
    height = 480
    preferred_fps = 30
    fps_mediatype_map = None # list of tuples in form of: (fps, media_type_index)
    
    readSetting = None
    _frame = None
    
    @property
    def name(self):
        return self.device.friendlyName
    
    @property
    def actual_width(self):
        return self.stream.listMediaType[self.deviceSettings.indexMediaType].width
    
    @property
    def actual_height(self):
        return self.stream.listMediaType[self.deviceSettings.indexMediaType].height 
    
    @property
    def src_id(self):
        return self.device.symbolicName
    
    def __init__(self, cam, size=(640,480), fps=None, timebase=None):
        self._init(cam, size, fps)
        
    def _init(self, cam, size=(640,480), fps=None):
        # setting up device
        self.device = cam.device
        self.deviceSettings = vi.DeviceSettings()
        self.deviceSettings.symbolicLink = self.device.symbolicName
        self.deviceSettings.indexStream = 0
        self.deviceSettings.indexMediaType = 0
        self.captureSettings = vi.CaptureSettings()
        self.captureSettings.readMode = vi.ReadMode.SYNC
        self.captureSettings.videoFormat = vi.CaptureVideoFormat.RGB24
        self.stream = self.device.listStream[self.deviceSettings.indexStream]
        
        # collecting additional information
        self.width = size[0]
        self.height = size[1]
        self.preferred_fps = fps
        self._initFrameRates()
        self._initMediaTypeId()
        
        self.context = _getVideoInputInstance()
        res = self.context.setupDevice(self.deviceSettings, self.captureSettings)
        if res != vi.ResultCode.OK:
            raise CameraCaptureError("Could not setup device. Error code: {:0f}".format(res))
    
        # creating frame buffer and initializing capture settings
        frame = np.empty((self.actual_height * self.actual_width * 3), dtype=np.uint8)
        self.readSetting = vi.ReadSetting()
        self.readSetting.symbolicLink = self.deviceSettings.symbolicLink
        self.readSetting.setNumpyArray(frame)
        frame.shape = (self.actual_height, self.actual_width, -1)
        self._frame = frame
        
        print (self.actual_height, self.actual_width, self.preferred_fps)
        logger.debug("Successfully set up device!")
        
    def re_init(self, cam, size=(640,480), fps=None):
        # TODO: close capture
        # TODO: set reset attributes
        self._init(cam, size, fps)
        
        # TODO: start new capture
    
    def get_frame(self):
        res = self.context.readPixels(self.readSetting)
        if res != vi.ResultCode.READINGPIXELS_DONE:
            print res
            raise CameraCaptureError("Could not receive frame. Error code: {:0f}".format(res))
        frame = Frame(self._frame)
        frame.timestamp = self.get_now()
        return frame
    
    @property
    def frame_rate(self):
        return self.stream.listMediaType[self.deviceSettings.indexMediaType].MF_MT_FRAME_RATE
    @frame_rate.setter
    def frame_rate(self, preferred_fps):
        self.preferred_fps = preferred_fps
        self._initMediaTypeId()
        
    def get_now(self):
        return time()

    @property
    def frame_size(self):
        raise Exception("Not implemented!")
    @frame_size.setter
    def frame_size(self, value):
        raise Exception("Not implemented!")
    
    def init_gui(self, sidebar):
        #TODO:
        pass
    
    def deinit_gui(self):
        # TODO:
        pass
    
    def close(self):
        self.deinit_gui()
        res = self.context.closeDevice(self.deviceSettings)
        if res != vi.ResultCode.OK:
            raise CameraCaptureError("Error while closing the capture device. Error code: %s" %res)
        
    def _initFrameRates(self):
        self.fps_mediatype_map = []
        media_types = self.stream.listMediaType
        for mt, i in zip(media_types, range(len(media_types))):
            if mt.width == self.width and mt.height == self.height:
                self.fps_mediatype_map.append((mt.MF_MT_FRAME_RATE, i))
        
        if not self.fps_mediatype_map:
            raise CameraCaptureError("Capture device does not support resolution: %d x %d"% (self.width, self.height))        
        
        logger.debug("found %d media types for given resolution" %len(self.fps_mediatype_map))
        self.fps_mediatype_map.sort()
        
    def _initMediaTypeId(self):
        match = None
        # choose highest framerate if none is given
        if self.preferred_fps is None:
            match = self.fps_mediatype_map[-1]
        else:
            # choose best match (fps +/- .25 is ok)
            for fps, i in self.fps_mediatype_map:
                if abs(fps-self.preferred_fps) < .25:
                    match = (fps, i)
                    break
            # if none is found, choose highest framerate
            if match is None:
                logger.warn("Capture device does not support preferred frame-rate %d"% (self.preferred_fps))
                match = self.fps_mediatype_map[-1]
        self.deviceSettings.indexMediaType = match[1]
        
def _getVideoInputInstance():
    return vi.videoInput_getInstance()






