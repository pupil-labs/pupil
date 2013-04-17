"""
uvc_capture is a module that build on opencv"s camera_capture
it adds some fuctionalty like:
    - access to all uvc controls
    - assosication by name patterns instead of id's (0,1,2..)
it requires:
    - opencv 2.3+
    - on Linux: v4l2-ctl (via apt-get install v4l2-util)
    - on MacOS: uvcc (binary is distributed with this module)
"""

from cv2 import VideoCapture, cvtColor, COLOR_RGB2BGR,COLOR_RGB2HSV, cv
import numpy as np
from os.path import isfile

import platform
os_name = platform.system()
del platform

if os_name == "Linux":
    import v4l2_ctl_oop as uvc
elif os_name == "Darwin":
    import uvcc as uvc

else:
    raise Exception("UVC Capture Error. No UVC Control backend found for this OS")

class CameraCapture(uvc.Camera):
    """docstring for Capture"""
    def __init__(self, cam,size=(640,480)):
        super(CameraCapture, self).__init__(cam)

        ###add cv videocapture capabilities
        self.cap = VideoCapture(self.cvId)
        self.set_size(size)
        self.read = self.cap.read

    def set_size(self,size):
        width,height = size
        self.cap.set(3, width)
        self.cap.set(4, height)

    def get_size(self):
        return self.cap.get(3),self.cap.get(4)

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

class noUVCCapture():
    """docstring for Capture"""
    def __init__(self, src,size=(640,480)):
        self.controls = None
        ###add cv videocapture capabilities
        self.cap = VideoCapture(src)
        self.set_size(size)
        self.read = self.cap.read

    def set_size(self,size):
        width,height = size
        self.cap.set(3, width)
        self.cap.set(4, height)

    def get_size(self):
        return self.cap.get(3),self.cap.get(4)

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

class FileCapture():
    """
    """
    def __init__(self,src):
        self.auto_rewind = True
        self.controls = None #No UVC controls available with file capture
        # we initialize the actual capture based on cv2.VideoCapture
        self.cap = VideoCapture(self.src)
        self._get_frame_ = self.cap.read


    def get_size(self):
        return self.cap.get(3),self.cap.get(4)

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



def autoCreateCapture(src,size=(640,480)):
        # checking src and handling all cases:
        src_type = type(src)

        #looking for attached cameras that match the suggested names
        if src_type is list:
            matching_devices = []
            for device in uvc.Camera_List():
                if any([s in device.name for s in src]):
                    matching_devices.append(device)

            if len(matching_devices) >1:
                print "Warning: found",len(matching_devices),"devices that match the src string pattern. Using the first one"
            if len(matching_devices) ==0:
                print "ERROR: No device found that matched",src,
                return

            cap = CameraCapture(matching_devices[0],size)
            print "camera selected: %s  with id: %s" %(cap.name,cap.cvId)
            return cap

        #looking for attached cameras that match cv_id
        elif src_type is int:
            for device in uvc.Camera_List():
                if device.cvId == src:
                    cap = CameraCapture(device,size)
                    print "camera selected: %s  with id: %s" %(cap.name,cap.cvId)
                    return cap

            #no matching id found or uvc control not supported: trying capture without uvc controls
            cap = noUVCCapture(src,size)
            print "no UVC support: using camera with id: %s" %(cap.name,cap.cvId)


        #looking for videofiles
        elif src_type is str:
            assert isfile(src),("autoCreateCapture: Could not locate", src)
            return FileCapture(src)
        else:
            raise Exception("autoCreateCapture: Could not create capture, wrong src_type")


if __name__ == '__main__':
    cap = autoCreateCapture(["Microsoft"],(1280,720))
    if cap:
        print cap.controls
        # cap.v4l2_set_default()
        # cap.uvc_camera.update_from_device()
    print "done"