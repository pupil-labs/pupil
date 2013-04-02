"""
uvc_capture is a module that build on opencv"s camera_capture
it adds some fuctionalty like:
    - access to all uvc controls
    - assosication by name patterns instead of id's (0,1,2..)
it requires:
    - opencv 2.3+
    - on linux: v4l2-ctl (via apt-get install v4l2-util)
    - on mac: uvcc (binary is distributed with this module)
"""

from cv2 import VideoCapture, cvtColor, COLOR_RGB2BGR,COLOR_RGB2HSV
import numpy as np
from os.path import isfile

import platform
os_name = platform.system()
del platform

if os_name == "Linux":
    import v4l2_ctl
    list_devices = v4l2_ctl.list_devices
elif os_name == "Darwin":

    def list_devices():
        return []


class Capture():
    """
    this is the capture base class
    src can be 3 things:
        - int: direclty assign a camera id
        - list of strings (can be just one):
            possible camera names that are supposed to be assinged
        - string: a path to a video file: load a video
    """
    def __init__(self,src,size=(640,480)):


        self.auto_rewind = False

        # checking src and handling all cases:
        src_type = type(src)

        if src_type is list:
            #looking for attached cameras that match the suggested names
            matching_devices = []
            for device in list_devices():
                if any([s in device["name"] for s in src]):
                    matching_devices.append(device)

            if len(matching_devices) >1:
                print "Warning: found",len(matching_devices),"devices that match the src string pattern. Using the first one"
            if len(matching_devices) ==0:
                print "ERROR: No device found that matched",src,
                self.cap = None
                self.src = None
                self._get_frame_= self._read_empty_
                return

            print "camera selected:", matching_devices[0]["name"], "id:",matching_devices[0]["src_id"]
            self.src = matching_devices[0]["src_id"]
            self.name = matching_devices[0]["name"]
        elif src_type is int:
            self.src = src
            self.name = "unnamed"
            for device in list_devices():
                if int(device["src_id"]) == src:
                    self.name = device["name"]
        elif src_type is str:
            if isfile(src):
                self.src = src
            else:
                print "ERROR could not find:",src
                self.src = None
                self.cap = None
                self._get_frame_= self._read_empty_
                return


        # we initialize the actual capture based on cv2.VideoCapture
        self.cap = VideoCapture(self.src)
        self._get_frame_ = self.cap.read

        if type(self.src) is int:
            #do all the uvc cpature relevant setup
            self.is_uvc_capture = True
            self.set_size(size)
            self.uvc_controls = self.extract_controls()

        else:
            #do all the vidoe file relevant setup
            self.is_uvc_capture = False
            self.uvc_controls = None



    def set_size(self,size):
        width,height = size
        self.cap.set(3, width)
        self.cap.set(4, height)

    def get_size(self):
        return self.cap.get(3),self.cap.get(4)

    def _read_empty_(self):
        """
        will be used when the capture init fails.
        """
        return False, None

    def read(self):
        s, img =self._get_frame_()
        if  self.auto_rewind and not s:
            self.rewind()
            s, img = self._get_frame_()
        return s,img

    def read_RGB(self):
        s,img = self.read()
        if s:
            cvtColor(img,COLOR_RGB2BGR,img)
        return s,img

    def read_HSV(self):
        s,img = self.read()
        if s:
            cvtColor(img,COLOR_RGB2HSV,img)
        return s,img

    def rewind(self):
        self.cap.set(1,0) #seek to the beginning


    if os_name == "Linux":

        def extract_controls(self):
            return v4l2_ctl.extract_controls(self.src)

        def uvc_get(self,control_name):
            device_id= self.src
            return v4l2_ctl.get(device_id,control_name)

        def uvc_set(self,value,control_name):
            device_id = self.src
            v4l2_ctl.set(device_id,control_name,value)

        def uvc_refresh_all(self):
            v4l2_ctl.update_from_device(self.uvc_controls)

        def uvc_set_default(self):
            for control in self.uvc_controls:
                self.uvc_set(self.uvc_controls[control]["default"],control)

    elif os_name == "Darwin":

        def extract_controls(self):
            return None

        def uvc_get(self,control_name):
            pass

        def uvc_set(self,value,control_name):
            pass

        def uvc_refresh_all(self):
            pass

        def uvc_set_default(self):
            pass

###these are special functions for the atb bar.
### you have to pass the specific control dict to to "data" in tw_add_var
### and name it the name of the control
if os_name == "Linux":
    atb_get = v4l2_ctl.getter
    atb_set = v4l2_ctl.setter
elif os_name == "Darwin":
    atb_get = None
    atb_set = None
else:
    atb_get = None
    atb_set = None

if __name__ == '__main__':
    cap = Capture(["C"],(1280,720))
    #s,img = cap.read()
    #print img.shape
    # print cap.uvc_controls
    # cap.v4l2_set_default()
    cap.uvc_refresh_all()
    print "done"