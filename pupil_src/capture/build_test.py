'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys,os
import objc
import QTKit
from QTKit import QTCaptureDevice,QTMediaTypeVideo
qt_cameras =  QTCaptureDevice.inputDevicesWithMediaType_(QTMediaTypeVideo)
print qt_cameras
import numpy
import cv2
import glfw
import c_methods
import atb
import OpenGL
import uvc_capture
print "HI QTKIT"

# print glfw.glfwInit()
print "goodbye"