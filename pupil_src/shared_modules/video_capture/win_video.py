'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import videoInput as vi
assert vi.VERSION >= 0.1
import numpy as np
import math
import cv2
from time import time, sleep
import logging
from videoInput import CaptureSettings, DeviceSettings
from fake_capture import Fake_Capture
from pyglui import ui

logger = logging.getLogger(__name__)

ERR_INIT_FAIL = "Could not setup capture device. "
MAX_RETRY_GRABBING_FRAMES = 5
MAX_RETRY_INIT_CAMERA = 5

class CameraCaptureError(Exception):
    """General Exception for this module"""
    def __init__(self, arg):
        super(CameraCaptureError, self).__init__()
        self.arg = arg


def device_list():
    devices = vi.DeviceList()
    _getVideoInputInstance().getListOfDevices(devices)

    cam_list = []
    for d in devices:
        cam_list.append({'name':d.friendlyName,'uid':d.symbolicName,'handle':d})
    return cam_list

class Frame(object):
    timestamp = 0
    width = 0
    height = 0

    _npy_frame = None
    _gray = None
    _bgr = None
    #indicate that the frame does not have a native yuv or jpeg buffer
    yuv_buffer = None
    jpeg_buffer = None

    def __init__(self, timestamp, npy_frame):
        self.timestamp = timestamp
        self._npy_frame = npy_frame
        self.height, self.width, _ = npy_frame.shape

    @property
    def img(self):
        return self.bgr

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self._npy_frame, cv2.COLOR_RGB2GRAY)
        return self._gray

    @property
    def bgr(self):
        if self._bgr is None:
            self._bgr = cv2.cvtColor(self._npy_frame, cv2.COLOR_BGRA2BGR)
        return self._bgr

class Camera_Capture(object):
    """
    Camera Capture encapsulates the videoInput class
    """
    deviceSettings = None
    captureSettings = None
    device = None
    stream = None

    menu = None
    sidebar = None
    width = 640
    height = 480
    preferred_fps = 30
    fps_mediatype_map = None # list of tuples in form of: (fps, media_type_index)

    readSetting = None
    _frame = None

    _is_initialized = False
    _failed_inits = 0

    @property
    def settings(self):
        settings = {}
        settings['name'] = self.name
        settings['frame_rate'] = self.frame_rate
        settings['frame_size'] = self.frame_size
        return settings
    @settings.setter
    def settings(self,settings):
        self.frame_size = settings['frame_size']
        self.frame_rate = settings['frame_rate']


    @property
    def name(self):
        if self.uid is not None:
            return str(self.device['name'])
        else:
            return "Fake Capture"

    @property
    def actual_width(self):
        if self.uid is not None:
            return self.stream.listMediaType[self.deviceSettings.indexMediaType].width
        else:
            return self.width

    @property
    def actual_height(self):
        if self.uid is not None:
            return self.stream.listMediaType[self.deviceSettings.indexMediaType].height
        else:
            return self.height

    @property
    def src_id(self):
        return self.uid

    def __init__(self, uid, size=(640,480), fps=None, timebase=None):
        self.init_capture(uid, size, fps)

    def re_init_capture(self, uid, size=(640,480), fps=None):
        if self.sidebar is None:
            self._close_device()
            self.init_capture(uid, size, fps)
        else:
            self.deinit_gui()
            self._close_device()
            self.init_capture(uid, size, fps)
            self.init_gui(self.sidebar)
            self.menu.collapsed = False

    def init_capture(self, uid, size=(640,480), fps=None, timebase=None):
        self.uid = uid
        if uid is not None:
            # validate parameter UID
            devices = device_list() # TODO: read list only once (initially) to save runtime
            for device in devices:
                print
                if device['uid'] == uid:
                    break
            if device['uid'] != uid:
                msg = ERR_INIT_FAIL + "UID of camera was not found."
                logger.error(msg)
                self.init_capture(None, size, fps, timebase)
                return

            # validate parameter SIZE
            if not len(size) == 2:
                msg = ERR_INIT_FAIL + "Parameter 'size' must have length 2."
                logger.error(msg)
                self.init_capture(None, size, fps, timebase)
                return

            # setting up device
            self.device = device
            self.deviceSettings = vi.DeviceSettings()
            self.deviceSettings.symbolicLink = self.device['uid']
            self.deviceSettings.indexStream = 0
            self.deviceSettings.indexMediaType = 0
            self.captureSettings = vi.CaptureSettings()
            self.captureSettings.readMode = vi.ReadMode.SYNC
            self.captureSettings.videoFormat = vi.CaptureVideoFormat.RGB32
            self.stream = self.device['handle'].listStream[self.deviceSettings.indexStream]

            # set timebase
            if timebase == None:
                logger.debug("Capture will run with default system timebase")
                self.timebase = 0
            else:
                logger.debug("Capture will run with app wide adjustable timebase")
                self.timebase = timebase

            # set properties
            self.width = size[0]
            self.height = size[1]
            self.preferred_fps = fps
            self._initFrameRates()
            self._initMediaTypeId()

            # robust camera initialization
            self.context = _getVideoInputInstance()
            while True:
                res = self.context.setupDevice(self.deviceSettings, self.captureSettings)
                if res != vi.ResultCode.OK:
                    self._failed_inits += 1
                    msg = ERR_INIT_FAIL + "Fall back to Fake Capture. Error code: %d" %(res)
                    if self._failed_inits < MAX_RETRY_INIT_CAMERA:
                        logger.warning("Retry initializing camera: {0}/{1}: ".format(self._failed_inits, MAX_RETRY_INIT_CAMERA) + msg)
                    else:
                        logger.error(msg)
                        self._failed_inits = 0
                        self.init_capture(None, size, fps, timebase)
                        return
                    sleep(0.25)
                else:
                    break

            # creating frame buffer and initializing capture settings
            frame = np.empty((self.actual_height * self.actual_width * 4), dtype=np.uint8)
            self.readSetting = vi.ReadSetting()
            self.readSetting.symbolicLink = self.deviceSettings.symbolicLink
            self.readSetting.setNumpyArray(frame)
            frame.shape = (self.actual_height, self.actual_width, -1)
            self._frame = frame

            logger.debug("Successfully set up device: %s @ %dx%dpx %dfps (mediatype %d)" %(self.name, self.actual_height, self.actual_width, self.frame_rate, self.deviceSettings.indexMediaType))
        else:
            self.device = Fake_Capture()
            self.width = size[0]
            self.height = size[1]
            self.preferred_fps = fps
        self._is_initialized = True
        self._failed_inits = 0

    def get_frame(self):
        if self.uid is not None:
            res = self.context.readPixels(self.readSetting)
            if res == vi.ResultCode.READINGPIXELS_REJECTED_TIMEOUT:
                for n in range(MAX_RETRY_GRABBING_FRAMES):
                    logger.warning("Retry reading frame: {0}/{1}. Error code: {2}".format(n+1, MAX_RETRY_GRABBING_FRAMES, res))
                    res = self.context.readPixels(self.readSetting)
                    if res == vi.ResultCode.READINGPIXELS_DONE:
                        break
            if res != vi.ResultCode.READINGPIXELS_DONE:
                msg = "Could not read frame. Fall back to Fake Capture. Error code: %d" %(res)
                logger.error(msg)
                self.re_init_capture(None, self.frame_size, self.preferred_fps)
                return self.get_frame()
            frame = Frame(self.get_now() - self.timebase, self._frame)
            return frame
        else:
            return self.device.get_frame_robust()

    @property
    def frame_rate(self):
        if self.uid is not None:
            return self.stream.listMediaType[self.deviceSettings.indexMediaType].MF_MT_FRAME_RATE
        else:
            return self.preferred_fps
    @frame_rate.setter
    def frame_rate(self, preferred_fps):
        self.re_init_capture(self.uid, (self.width, self.height), preferred_fps)

    @property
    def available_frame_rates(self):
        fps_list = []
        for fps, _ in self.fps_mediatype_map:
            fps_list.append(fps)
        return fps_list

    @property
    def frame_size(self):
        return (self.actual_width, self.actual_height)
    @frame_size.setter
    def frame_size(self, size):
        self.re_init_capture( self.uid, size, self.preferred_fps)

    @property
    def available_frame_sizes(self):
        size_list = []
        for size in self.size_mediatype_map:
            size_list.append(size)
        return size_list

    @property
    def jpeg_support(self):
        return False

    def get_now(self):
        return time()

    def get_timestamp():
        return self.get_now()-self.timebase.value

    def init_gui(self,sidebar):

        def gui_init_cam(uid):
            logger.debug("selected new device: " + str(uid))
            self.re_init_capture(uid, (self.width, self.height), self.preferred_fps)

        def gui_get_cam():
            return self.uid

        def gui_get_frame_size():
            return self.frame_size

        def gui_set_frame_size(new_size):
            self.frame_size = new_size

        def gui_get_frame_rate():
            return self.frame_rate

        def gui_set_frame_rate(new_fps):
            self.frame_rate = new_fps

        #create the menu entry
        self.menu = ui.Growing_Menu(label='Camera Settings')

        self.menu.append(ui.Info_Text("Device: " + self.name))
        # TODO: refresh button for capture list. Make properties that refresh on reading...
        cams = device_list()
        cam_names = ['Fake Capture'] + [str(c["name"]) for c in cams]
        cam_devices = [None] + [c["uid"] for c in cams]
        self.menu.append(ui.Selector('device',self,selection=cam_devices,labels=cam_names,label='Capture Device', getter=gui_get_cam, setter=gui_init_cam))

        #hardware_ts_switch = ui.Switch('use_hw_ts',self,label='use hardware timestamps')
        #hardware_ts_switch.read_only = True
        #self.menu.append(hardware_ts_switch)

        #self.menu.append(ui.Selector('frame_size', selection=self.available_frame_sizes, label='Frame Size', getter=gui_get_frame_size, setter=gui_set_frame_size))
        if self.uid is not None:
            self.menu.append(ui.Info_Text("Resolution: {0} x {1} pixels".format(self.actual_width, self.actual_height)))
            self.menu.append(ui.Selector('frame_rate', selection=self.available_frame_rates, label='Frame Rate', getter=gui_get_frame_rate, setter=gui_set_frame_rate))

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
        self._close_device()

    def _close_device(self):
        if self.uid is None:
            return
        if self._is_initialized:
            self._is_initialized = False
            res = self.context.closeDevice(self.deviceSettings)
            if res != vi.ResultCode.OK:
                msg = "Error while closing the capture device. Error code: %s" %res
                logger.error(msg)
                raise CameraCaptureError(msg)

    def _initFrameRates(self):
        """ Reads out possible frame-rates for a given resolution and stores result as internal frame-rate map. """
        self.fps_mediatype_map = []
        tmp_fps_values = []
        self.size_mediatype_map = []
        media_types = self.stream.listMediaType
        for mt, i in zip(media_types, range(len(media_types))):
            size_tuple = (mt.width, mt.height)
            # add distinct resolutions
            if not size_tuple in self.size_mediatype_map:
                self.size_mediatype_map.append(size_tuple)
            if mt.width == self.width and mt.height == self.height:
                # add distinct frame-rate options
                if not mt.MF_MT_FRAME_RATE in tmp_fps_values:
                    tmp_fps_values.append(mt.MF_MT_FRAME_RATE)
                    self.fps_mediatype_map.append((mt.MF_MT_FRAME_RATE, i))

        if not self.fps_mediatype_map:
            msg = ERR_INIT_FAIL + "Capture device does not support resolution: %d x %d"% (self.width, self.height)
            logger.error(msg)
            raise CameraCaptureError(msg)

        logger.debug("found %d media types for given resolution: %s" %(len(self.fps_mediatype_map), str(self.fps_mediatype_map)))
        self.fps_mediatype_map.sort()

    def _initMediaTypeId(self):
        """ Selects device by setting media-type ID based on previously initialized frame-rate map. """
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
    """ Returns an instance of the videoInput class. """
    return vi.videoInput_getInstance()