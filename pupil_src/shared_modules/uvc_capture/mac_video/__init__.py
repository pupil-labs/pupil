'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License. 
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
OOP style interface for uvcc c_types binding

Three classes:
    Camera_List holds Cam's instances,
    Cam is a class that contains infos about attached cameras
    Camera  get initialized with a Cam instance it holds each device handle, names, controls ect.
    Control is the actual Control with methods for getting and setting them.
"""
from cv2 import VideoCapture
from time import time
import QTKit
from raw import *

class Control(object):
    """docstring for uvcc_Control"""
    def __init__(self,name,i,handle):
        self.handle = handle
        self.name = name
        self.atb_name = name[9:].capitalize() #pretify the name
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
        self.value = val
        return uvccSetVal(val,self.name,self.handle)

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
        print "UVCC released"
        uvccReleaseCam(self.handle)
        uvccExit()


class Frame(object):
    """docstring of Frame"""
    def __init__(self, timestamp,img,compressed_img=None, compressed_pix_fmt=None):
        self.timestamp = timestamp
        self.img = img 
        self.compressed_img = compressed_img
        self.compressed_pix_fmt = compressed_pix_fmt


class Camera_Capture(object):
    """docstring for uvcc_camera"""
    def __init__(self, cam,size=(640,480),fps=30):
        self.src_id = cam.src_id
        self.uId = cam.uId
        self.name = cam.name
        self.controls = Controls(self.uId)
        self.capture = VideoCapture(self.src_id)
        self.set_size(size)

    def get_frame(self):
        s, img = self.capture.read()
        timestamp = time()
        return Frame(timestamp,img)

    def set_size(self,size):
        width,height = size
        self.capture.set(3, width)
        self.capture.set(4, height)

    def get_size(self):
        return self.capture.get(3), self.capture.get(4)

    def set_fps(self,fps):
        self.capture.set(5,fps)

    def get_fps(self):
        return self.capture.get(5)

    def make_atb_bar(self,pos):
        return pos

    def kill_atb_bar(self,pos):
        return pos



class Cam():
    """a simple class that only contains info about a camera"""
    def __init__(self,name,uId,src_id):
        self.src_id = src_id
        self.uId = uId
        self.name = name

class Camera_List(list):
    """docstring for uvcc_control"""

    def __init__(self):
        # QTCaptureDevice inputDevicesWithMediaType:QTMediaTypeVideo
        qt_cameras =  QTKit.QTCaptureDevice.inputDevicesWithMediaType_(QTKit.QTMediaTypeVideo)
        for src_id,q in enumerate(qt_cameras):
            uId =  q.uniqueID()
            name = q.localizedDisplayName().encode('utf-8')
            self.append(Cam(name,uId,src_id))


if __name__ == '__main__':
    # import cv2
    # _ = cv2.VideoCapture(-1) # we can to wake the isight camera up if we want to query more information....
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
