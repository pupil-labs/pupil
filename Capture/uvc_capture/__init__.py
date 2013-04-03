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
    import v4l2_ctl_oop as uvc
elif os_name == "Darwin":
    import uvcc as uvc


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
        if src_type is not str: #we are looking for an actual camera not a video file...
            self.uvc_camera_list = uvc.Camera_List()

            if src_type is list:
                #looking for attached cameras that match the suggested names
                matching_devices = []
                for device in self.uvc_camera_list:
                    if any([s in device.name for s in src]):
                        matching_devices.append(device)

                if len(matching_devices) >1:
                    print "Warning: found",len(matching_devices),"devices that match the src string pattern. Using the first one"
                if len(matching_devices) ==0:
                    print "ERROR: No device found that matched",src,
                    self.cap = None
                    self.src = None
                    self._get_frame_= self._read_empty_
                    uvc_camera_list.release()
                    return

                self.uvc_camera = matching_devices[0]
                print "camera selected:", self.uvc_camera.name, "id:",self.uvc_camera.cv_id
                self.name = self.uvc_camera.name
                self.src = self.uvc_camera.cv_id

            elif src_type is int:
                self.src = src
                self.name = "unnamed"
                for device in self.uvc_camera_list:
                    if int(device.cv_id) == src:
                        self.uvc_camera = device
                        print "camera selected:", self.uvc_camera.name, "id:",self.uvc_camera.cv_id
                        self.name = self.uvc_camera.name
                        self.src = self.uvc_camera.cv_id


            #do all the uvc cpature relevant setup
            self.cap = VideoCapture(self.src)
            self.set_size(size)
            self._get_frame_ = self.cap.read
            self.uvc_camera.init_controls()


        ###setup as video playback
        if src_type is str:
            if isfile(src):
                self.src = src
                # we initialize the actual capture based on cv2.VideoCapture
                self.cap = VideoCapture(self.src)
                self._get_frame_ = self.cap.read
                #do all the vidoe file relevant setup
                self.uvc_camera = None
            else:
                print "ERROR could not find:",src
                self.src = None
                self.cap = None
                self._get_frame_= self._read_empty_
                return


    def release(self):
        print "Cleaned up uvc_control."
        try:
            self.uvc_camera_list.release()
        except:
            pass

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
        pass
    elif os_name == "Darwin":
        pass


if __name__ == '__main__':
    cap = Capture(["525"],(1280,720))
    s,img = cap.read()
    #print img.shape
    # print cap.uvc_controls
    # cap.v4l2_set_default()
    cap.uvc_camera.refresh_all()
    cap.release()
    print "done"