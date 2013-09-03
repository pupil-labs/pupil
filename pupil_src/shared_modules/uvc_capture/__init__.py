'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License. 
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
uvc_capture is a module that extends opencv"s camera_capture for mac and windows
on Linux it repleaces it completelty.
it adds some fuctionalty like:
    - access to all uvc controls
    - assosication by name patterns instead of id's (0,1,2..)
it requires:
    - opencv 2.3+
    - on Linux: v4l2-ctl (via apt-get install v4l2-util)
    - on MacOS: uvcc (binary is distributed with this module)
"""

from cv2 import VideoCapture
import numpy as np
from os.path import isfile

import platform
os_name = platform.system()
del platform


###OS specific imports and defs
if os_name == "Linux":
    from linux_video import Camera_Capture,Camera_List
elif os_name == "Darwin":
    from mac_video import Camera_Capture,Camera_List
else:
    from other_video import Camera_Capture,Camera_List


# non os specific defines
class FileCapture():
    """
    simple file capture that can auto_rewind
    """
    def __init__(self,src):
        self.auto_rewind = True
        self.controls = None #No UVC controls available with file capture
        # we initialize the actual capture based on cv2.VideoCapture
        self.cap = VideoCapture(src)
        self._get_frame_ = self.cap.read


    def get_size(self):
        return self.cap.get(3),self.cap.get(4)

    def set_fps(self):
        pass

    def get_fps(self):
        return None

    def read(self):
        s, img =self._get_frame_()
        if  self.auto_rewind and not s:
            self.rewind()
            s, img = self._get_frame_()
        return s,img


    def rewind(self):
        self.cap.set(1,0) #seek to the beginning



def autoCreateCapture(src,size=(640,480),fps=30):
    # checking src and handling all cases:
    src_type = type(src)

    #looking for attached cameras that match the suggested names
    if src_type is list:
        matching_devices = []
        for device in Camera_List():
            if any([s in device.name for s in src]):
                matching_devices.append(device)

        if len(matching_devices) >1:
            print "Warning: found",len(matching_devices),"devices that match the src string pattern. Using the first one."
        if len(matching_devices) ==0:
            print "ERROR: No device found that matched",src,
            return

        cap = Camera_Capture(matching_devices[0],size,fps)
        print "camera selected: %s  with id: %s" %(cap.name,cap.src_id)
        return cap

    #looking for attached cameras that match cv_id
    elif src_type is int:
        for device in Camera_List():
            if device.src_id == src:
                cap = Camera_Capture(device,size,fps)
                print "camera selected: %s  with id: %s" %(cap.name,cap.src_id)
                return cap

        #control not supported: trying capture without uvc controls
        cap = Camera_Capture(src,size,fps)
        print "WARNING: no UVC support: Using camera with id: %s" %(src)
        return cap


    #looking for videofiles
    elif src_type is str:
        assert isfile(src),("autoCreateCapture: Could not locate VideoFile:", src)
        print "Using video file as source ",src
        return FileCapture(src)
    else:
        raise Exception("autoCreateCapture: Could not create capture, wrong src_type")


if __name__ == '__main__':
    cap = autoCreateCapture(1,(1280,720),30)
    if cap:
        print cap.controls
    print "done"