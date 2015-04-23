'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
OOP style interface for uvcc c_types binding and wrapper for cv2 videocapture

Three classes:
    Camera_List holds Cam's instances,
    Cam is a class that contains infos about attached cameras
    Camera  get initialized with a Cam instance it holds each device handle, names, controls ect.
    Control is the actual Control with methods for getting and setting them.
"""
import sys
from pyglui import ui
from time import time
from raw import *
import cv2

#logging
import logging
logger = logging.getLogger(__name__)


class CameraCaptureError(Exception):
    """General Exception for this module"""
    pass



class Frame(object):
    """docstring of Frame"""
    def __init__(self, timestamp,img):
        self.timestamp = timestamp
        self.img = img
        self.height,self.width,_ = img.shape
        self._gray = None
        self._yuv = None

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        return self._gray




class Camera_Capture(object):
    """docstring for uvcc_camera"""
    def __init__(self, cam,size=(640,480),fps=30,timebase=None):
        self.fps = 30
        self.src_id = cam.src_id
        self.uId = cam.uId
        self.name = cam.name
        self.controls = Controls(self.uId)

        self.sidebar = None # this holds a pointer to the app gui used in init_gui
        self.menu = None

        if timebase == None:
            logger.debug("Capture will run with default system timebase")
            self.timebase = c_double(0)
        elif hasattr(timebase,'value'):
            logger.debug("Capture will run with app wide adjustable timebase")
            self.timebase = timebase
        else:
            logger.error("Invalid timebase variable type. Will use default system timebase")
            self.timebase = c_double(0)

        try:
            self.controls['UVCC_REQ_FOCUS_AUTO'].set_val(0)
        except KeyError:
            pass

        self.capture = cv2.VideoCapture(self.src_id)
        self.frame_size = size
        self.frame_rate = fps


    def re_init(self,cam,size=(640,480),fps=30):
        self.src_id = cam.src_id
        self.uId = cam.uId
        self.name = cam.name
        self.controls = Controls(self.uId)

        try:
            self.controls['UVCC_REQ_FOCUS_AUTO'].set_val(0)
        except KeyError:
            pass

        self.capture = cv2.VideoCapture(self.src_id)
        self.frame_size = size
        self.frame_rate = fps

        #recreate the gui with new values
        self.deinit_gui()
        self.init_gui(self.sidebar)

    def re_init_cam_by_src_id(self,src_id):
        try:
            cam = Camera_List()[src_id]
        except KeyError:
            logger.warning("could not reinit capture, src_id not valid anymore")
            return
        self.re_init(cam,self.frame_size)

    def get_frame(self):
        s, img = self.capture.read()
        if not s:
            raise CameraCaptureError("Could not get frame")
        timestamp = time()-self.timebase.value
        return Frame(timestamp,img)


    @property
    def frame_size(self):
        return self.capture.get(3), self.capture.get(4)
    @frame_size.setter
    def frame_size(self,size):
        width,height = size
        self.capture.set(3, width)
        self.capture.set(4, height)

    @property
    def frame_rate(self):
        fps = self.capture.get(5)
        if fps != 0:
            return fps
        else:
            return self.fps
    @frame_rate.setter
    def frame_rate(self,fps):
        self.capture.set(5,fps)


    def get_now(self):
        return time()

    def init_gui(self,sidebar):


        sorted_controls = [c for c in self.controls.itervalues()]
        sorted_controls.sort(key=lambda c: c.order)


        self.menu = ui.Growing_Menu(label='Camera Settings')


        cameras = Camera_List()
        camera_names = [c.name for c in cameras]
        camera_ids = [c.src_id for c in cameras]
        self.menu.append(ui.Selector('src_id',self,selection=camera_ids,labels=camera_names,label='Capture Device', setter=self.re_init_cam_by_src_id) )

        hardware_ts_switch = ui.Switch('hardware_timestamps',None,getter=lambda:False,label='use hardware timestamps')
        hardware_ts_switch.read_only=True
        self.menu.append(hardware_ts_switch)

        for control in sorted_controls:
            name = control.pretty_name
            c = None
            if control.type=="bool":
                c = ui.Switch('value',control,setter=control.set_val,label=name)
            elif control.type=='int':
                c = ui.Slider('value',control,min=control.min,max=control.max,
                                step=control.step, setter=control.set_val,label=name)

            elif control.type=="menu":
                if control.menu is None:
                    selection = range(control.min,control.max+1,control.step)
                    labels = selection
                else:
                    #this is currenlty not implemented
                    selection = [c.val for c in control.menu]
                    labels = [c.name for c in control.menu]
                c = ui.Selector('value',control,selection=selection,labels = labels,label=name,setter=control.set_val)
            else:
                pass
                # print control.type
            # if control.flags == "inactive":
                # c.read_only = True
            if c is not None:
                self.menu.append(c)

        self.menu.append(ui.Button("refresh",self.controls.update_from_device))
        self.menu.append(ui.Button("load defaults",self.controls.load_defaults))
        self.sidebar = sidebar
        #add below geneal settings
        self.sidebar.insert(1,self.menu)

    def deinit_gui(self):
        if self.menu:
            self.sidebar.remove(self.menu)
            self.menu = None

    def close(self):
        self.control = None
        logger.info("Capture released")
        pass


class Control(object):
    """docstring for uvcc_Control"""
    def __init__(self,name,i,handle):
        self.handle = handle
        self.name = name
        self.pretty_name = name[9:].capitalize() #pretify the name
        self.order = i
        self.value = None
        self.assess_type()


    def assess_type(self):
        """
        find out if a control is active
        find out if the range is bool or int
        """
        self.value = None
        self.min = None
        self.max = None
        self.step    = None
        self.default = None
        self.menu    = None #some day in the future we will extract the control menu entries here.

        self.info = self.get_info()
        """
        D0 1 = Supports GET value requests      Capability
        D1 1 = Supports SET value requests      Capability
        D2 1 = Disabled due to automatic mode (under device control)    State
        D3 1 = Autoupdate Control   Capability
        D4 1 = Asynchronous Control Capability
        D5 1 = Disabled due to incompatibility with Commit state.   State
        """
        if self.info > 0 :  # Control supported
            self.value = self.get_val_from_device()
            self.min =  self.get_(UVC_GET_MIN)
            self.max =  self.get_(UVC_GET_MAX)
            self.step    =  self.get_(UVC_GET_RES)
            self.default =  self.get_(UVC_GET_DEF)

            if ((self.max,self.min) == (None,None)) or ((self.max,self.min) == (1,0)) :
                self.type  = "bool"
            # elif (self.max,self.min) == (None,None):
            #     ###I guess this should be a menu
            #     self.type  = "int"
            #     self.flags = "active"
            #     self.min = 0
            #     self.max = 20
            #     self.step = 1
            else:
                self.type  = "int"

            if self.info >> 3 & 1: # Disabled due to automatic mode (under device control)
                self.flags = "inactive"
            else:
                self.flags = "active"
        else:
            self.type  = "unknown type"
            self.flags = "control not supported"
            self.value = None

    def get_val_from_device(self):
        return uvccGetVal(self.name,self.handle)

    def get_val(self):
        return self.value

    def set_val(self,val):
        uvccSetVal(val,self.name,self.handle)
        new_val = uvccGetVal(self.name,self.handle)
        if new_val is not None:
            self.value = uvccGetVal(self.name,self.handle)

    def get_info(self):
        return uvccRequestInfo(self.name,self.handle)

    def get_(self,request):
        return uvccSendRequest(self.name,request,self.handle)


class Controls(dict):
    """docstring for Controls"""
    def __init__(self,uId):
        uvccInit()
        self.handle = uvccGetCamWithQTUniqueID(uId)
        assert self.handle is not None, "UVCC could not open camera based on uId %s" %uId
         # list of all controls implemented by uvcc,
         # the names evaluate to ints using a dict lookup in raw.py
        controls_str = uvcc_controls[:-1] #the last one is not a real control
        for i,c in enumerate(controls_str):
            self[c] = Control(c,i,self.handle)

    def update_from_device(self):
        for c in self.itervalues():
            if c.flags == "active":
                c.value = c.get_val_from_device()

    def load_defaults(self):
        for c in self.itervalues():
            if c.flags == "active" and c.default is not None:
                c.set_val(c.default)

    def __del__(self):
        uvccReleaseCam(self.handle)
        uvccExit()


class Cam():
    """a simple class that only contains info about a camera"""
    def __init__(self,name,uId,src_id):
        self.src_id = src_id
        self.uId = uId
        self.name = name

class Camera_List(list):
    """docstring for uvcc_control"""

    def __init__(self):
        if getattr(sys, 'frozen', False):
            #explicit import needed when frozen
            import QTKit

        from QTKit import QTCaptureDevice,QTMediaTypeVideo
        qt_cameras =  QTCaptureDevice.inputDevicesWithMediaType_(QTMediaTypeVideo)
        for src_id,q in enumerate(qt_cameras):
            uId =  q.uniqueID()
            name = q.localizedDisplayName().encode('utf-8')
            self.append(Cam(name,uId,src_id))

if __name__ == '__main__':
    # import cv2
    # _ = cv2.cv2.VideoCapture(-1) # we can to wake the isight camera up if we want to query more information....
    uvc_cameras = Camera_List()
    for cam in uvc_cameras:
        print cam.name
        print cam.src_id
        print cam.uId
    # camera = Camera(uvc_cameras[1])

    # print camera.name
    # #     cam.init_controls()
    # #     cam.load_defaults()
    # for c in camera.controls.itervalues():
    #     if c.flags != "control not supported":
    #         print c.name, " "*(40-len(c.name)), c.value, c.min,c.max,c.step
