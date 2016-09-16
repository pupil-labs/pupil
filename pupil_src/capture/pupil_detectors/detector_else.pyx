'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from methods import  normalize
from pyglui import ui

import cv2

cdef extern from '<opencv2/core/types_c.h>':

  int CV_8UC1
  int CV_8UC3

cdef extern from '<opencv2/core/core.hpp>' namespace 'cv':

    cdef cppclass Mat :
        Mat() except +
        Mat( int height, int width, int type, void* data  ) except+
        Mat( int height, int width, int type ) except+

    cdef cppclass Point2f :
        float x
        float y

    cdef cppclass Size2f :
        float height
        float width

    cdef cppclass RotatedRect :
        float angle
        Point2f center
        Size2f size

cdef extern from "algo.cpp" namespace 'ELSE':

    cdef RotatedRect run(Mat input_img)

cdef inline convertToElSePythonResult( RotatedRect& result, object frame):

    ellipse = {}
    ellipse['center'] = (result.center.x,result.center.y)
    ellipse['axes'] =  (result.size.width,result.size.height)
    ellipse['angle'] = result.angle

    py_result = {}
    py_result['confidence'] = .8
    py_result['ellipse'] = ellipse
    py_result['diameter'] = max(ellipse['axes'])

    norm_center = normalize( ellipse['center'] , (frame.width, frame.height),flip_y=True)
    py_result['norm_pos'] = norm_center
    py_result['timestamp'] = frame.timestamp
    py_result['method'] = '2d ElSe'

    return py_result

cdef inline convertToElSePythonResult2(object frame):

    ellipse = {}
    ellipse['center'] = (0.,0.)
    ellipse['axes'] =  (10.,10.)
    ellipse['angle'] = 10.

    py_result = {}
    py_result['confidence'] = .8
    py_result['ellipse'] = ellipse
    py_result['diameter'] = max(ellipse['axes'])

    norm_center = normalize( ellipse['center'] , (frame.width, frame.height),flip_y=True)
    py_result['norm_pos'] = norm_center
    py_result['timestamp'] = frame.timestamp
    py_result['method'] = '2d ElSe Fake'

    return py_result


cdef class Detector_ElSe:

    cdef object menu
    cdef object gPool

    def __cinit__(self,*args,**kwargs):
        pass
    def __init__(self, g_pool = None, *args,**kwargs):
        self.gPool = g_pool

    def detect(self, frame_, *args, **kwargs):
        image_width = frame_.width
        image_height = frame_.height

        #cdef unsigned char[:,::1] img = cv2.resize(frame_.gray, (384,288))
        cdef unsigned char[:,::1] img = frame_.gray

        cdef Mat frame = Mat(image_height, image_width, CV_8UC1, <void *> &img[0,0] )

        cdef RotatedRect rot_rect = run(frame) # might need `deref`
        py_result = convertToElSePythonResult( rot_rect, frame_)
        #py_result = convertToElSePythonResult2(frame_)
        print '%s'%py_result

        return py_result

    def init_gui(self,sidebar):
        self.menu = ui.Growing_Menu('Pupil Detector')
        info = ui.Info_Text("Uses ElSe (algorithmic split) algorithm.")
        self.menu.append(info)
        sidebar.append(self.menu)

    def deinit_gui(self):
        self.gPool.sidebar.remove(self.menu)
        self.menu = None

    def cleanup(self):
        self.deinit_gui()

    def visualize(self):
        pass
