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
import cv2
from time import time
import logging
from videoInput import CaptureSettings, DeviceSettings
from pyglui import ui

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
        
        # TODO: property bus_info

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

    def __init__(self, timestamp, npy_frame):
        self.timestamp = timestamp
        self._npy_frame = npy_frame
        self.height, self.width, _ = npy_frame.shape

    @property
    def img(self):
        return cv2.cvtColor(self._npy_frame, cv2.COLOR_RGBA2RGB)

    @property
    def gray(self):
        return cv2.cvtColor(self._npy_frame, cv2.COLOR_RGB2GRAY)

    @property
    def bgr(self):
        return cv2.cvtColor(self._npy_frame, cv2.COLOR_RGB2BGR)

class Camera_Capture(object):
    """
    Camera Capture encapsulates the videoInput class
    """
    deviceSettings = None
    captureSettings = None
    device = None
    stream = None

    menu = None
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

    def _init(self, cam, size=(640,480), fps=None, timebase=None):
        # setting up device
        self.device = cam.device
        self.deviceSettings = vi.DeviceSettings()
        self.deviceSettings.symbolicLink = self.device.symbolicName
        self.deviceSettings.indexStream = 0
        self.deviceSettings.indexMediaType = 0
        self.captureSettings = vi.CaptureSettings()
        self.captureSettings.readMode = vi.ReadMode.SYNC
        self.captureSettings.videoFormat = vi.CaptureVideoFormat.RGB32
        self.stream = self.device.listStream[self.deviceSettings.indexStream]

        # collecting additional information
        if timebase == None:
            logger.debug("Capture will run with default system timebase")
            self.timebase = 0
        else:
            logger.debug("Capture will run with app wide adjustable timebase")
            self.timebase = timebase
        
        self.width = size[0]
        self.height = size[1]
        self.preferred_fps = fps
        self._initFrameRates()
        self._initMediaTypeId()

        self.context = _getVideoInputInstance()
        res = self.context.setupDevice(self.deviceSettings, self.captureSettings)
        if res != vi.ResultCode.OK:
            raise CameraCaptureError("Could not setup device. Error code: %d" %(res))

        # creating frame buffer and initializing capture settings
        frame = np.empty((self.actual_height * self.actual_width * 4), dtype=np.uint8)
        self.readSetting = vi.ReadSetting()
        self.readSetting.symbolicLink = self.deviceSettings.symbolicLink
        self.readSetting.setNumpyArray(frame)
        frame.shape = (self.actual_height, self.actual_width, -1)
        self._frame = frame

        logger.debug("Successfully set up device: %s @ %dx%dpx %dfps (mediatype %d)" %(self.name, self.actual_height, self.actual_width, self.frame_rate, self.deviceSettings.indexMediaType))

    def re_init(self, cam, size=(640,480), fps=None):
        # TODO: close capture
        # TODO: set reset attributes
        self._init(cam, size, fps)

        # TODO: start new capture

    def get_frame(self):
        res = self.context.readPixels(self.readSetting)
        if res == vi.ResultCode.READINGPIXELS_REJECTED_TIMEOUT:
            for n in range(5):
                logger.warning("Reading frame timed out, retry %d/5" %(n+1))
                res = self.context.readPixels(self.readSetting)
                if res == vi.ResultCode.READINGPIXELS_DONE:
                    break
        if res != vi.ResultCode.READINGPIXELS_DONE:
            msg = "Could not receive frame. Error code: %d" %(res)
            logger.error(msg)
            raise CameraCaptureError(msg)
        frame = Frame(self.get_now() - self.timebase, self._frame)
        return frame

    @property
    def frame_rate(self):
        return self.stream.listMediaType[self.deviceSettings.indexMediaType].MF_MT_FRAME_RATE
    @frame_rate.setter
    def frame_rate(self, preferred_fps):
        self.preferred_fps = preferred_fps
        self._initMediaTypeId()

    @property
    def frame_size(self):
        return (self.actual_width, self.actual_height)
    @frame_size.setter
    def frame_size(self, value):
        raise Exception("Not implemented!")

    def get_now(self):
        return time()

    def init_gui(self,sidebar):

        #create the menu entry
        self.menu = ui.Growing_Menu(label='Camera Settings')
        # cameras = Camera_List()
        # camera_names = [c.name for c in cameras]
        # camera_ids = [c.src_id for c in cameras]
        # self.menu.append(ui.Selector('src_id',self,selection=camera_ids,labels=camera_names,label='Capture Device', setter=gui_init_cam_by_src_id) )

        #hardware_ts_switch = ui.Switch('use_hw_ts',self,label='use hardware timestamps')
        #hardware_ts_switch.read_only = True
        #self.menu.append(hardware_ts_switch)

        # self.menu.append(ui.Selector('frame_rate', selection=self.capture.frame_rates,labels=[str(d/float(n)) for n,d in self.capture.frame_rates],
        #                                 label='Frame Rate', getter=gui_get_frame_rate, setter=gui_set_frame_rate) )


        # for control in self.controls:
        #     c = None
        #     ctl_name = control['name']

        #     # we use closures as setters and getters for each control element
        #     def make_setter(control):
        #         def fn(val):
        #             self.capture.set_control(control['id'],val)
        #             control['value'] = self.capture.get_control(control['id'])
        #         return fn
        #     def make_getter(control):
        #         def fn():
        #             return control['value']
        #         return fn
        #     set_ctl = make_setter(control)
        #     get_ctl = make_getter(control)

        #     #now we add controls
        #     if control['type']=='bool':
        #         c = ui.Switch(ctl_name,getter=get_ctl,setter=set_ctl)
        #     elif control['type']=='int':
        #         c = ui.Slider(ctl_name,getter=get_ctl,min=control['min'],max=control['max'],
        #                         step=control['step'], setter=set_ctl)

        #     elif control['type']=="menu":
        #         if control['menu'] is None:
        #             selection = range(control['min'],control['max']+1,control['step'])
        #             labels = selection
        #         else:
        #             selection = [value for name,value in control['menu'].iteritems()]
        #             labels = [name for name,value in control['menu'].iteritems()]
        #         c = ui.Selector(ctl_name,getter=get_ctl,selection=selection,labels = labels,setter=set_ctl)
        #     else:
        #         pass
        #     if control['disabled']:
        #         c.read_only = True
        #     if ctl_name == 'Exposure, Auto Priority':
        #         # the controll should always be off. we set it to 0 on init (see above)
        #         c.read_only = True

        #     if c is not None:
        #         self.menu.append(c)

        # self.menu.append(ui.Button("refresh",gui_update_from_device))
        # self.menu.append(ui.Button("load defaults",gui_load_defaults))
        self.menu.collapsed = True
        self.sidebar = sidebar
        #add below geneal settings
        self.sidebar.insert(1,self.menu)

    def deinit_gui(self):
        if self.menu:
            self.sidebar.remove(self.menu)
            self.menu = None

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

        logger.debug("found %d media types for given resolution: %s" %(len(self.fps_mediatype_map), str(self.fps_mediatype_map)))
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
