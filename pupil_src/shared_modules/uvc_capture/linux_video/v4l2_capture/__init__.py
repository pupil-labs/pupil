'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
This file contains bindings to a webcam capture module that works with v4l2 and is written in c
"""

from ctypes import *
from numpy.ctypeslib import ndpointer
import numpy as np
import os

### Get location of  this file
source_loc = os.path.dirname(os.path.abspath(__file__))

### Run Autocompiler
#  Binaries are not distributed instead a make file and source are in this folder
#  Make is invoked when this module is imported or run.

arch_64bit = sizeof(c_void_p) == 8
if arch_64bit:
    c_flags = "CFLAGS=-m64"
else:
    c_flags = "CFLAGS=-m32"

from subprocess import check_output
# print " compiling now."
compiler_status = check_output(["make",c_flags],cwd=source_loc)
# print compiler_status
del check_output
# print "c-methods: compiling done."


### C-Types binary loading
dll_name = "capture.so"
dllabspath = source_loc + os.path.sep + dll_name
if not os.path.isfile(dllabspath):
    raise Exception("v4l2 capture Error could not find binary.")

dll = CDLL(dllabspath)


from definitions import *

### C-Types Argtypes and Restype

dll.open_device.argtypes = [c_char_p,]
dll.open_device.restype = c_void_p
dll.close_device.argtypes = [c_void_p,]
dll.close_device.restype = c_int
dll.get_buffer.argtypes = [c_int,POINTER(v4l2_buffer)]
dll.get_buffer.restype = c_void_p
dll.release_buffer.restype = c_int
dll.init_device.argtypes = [c_int,POINTER(c_uint32),POINTER(c_uint32),POINTER(c_uint32)]


# open device
# setup fmt
# create and map buffers
# qbuffers and start stream
# dque buffer and return POINTER
# qbuffer repead above
# stop stream
# umap buffer

class Frame(object):
    """docstring for Frame"""
    def __init__(self, timestamp,img,compressed_img=None, compressed_pix_fmt=None):
        self.timestamp = timestamp
        self.img = img 
        self.compressed_img = compressed_img
        self.compressed_pix_fmt = compressed_pix_fmt


class VideoCapture(object):
    """docstring for v4l2_capture"""
    def __init__(self, src_id,size=(1280,720),fps=24):
        if src_id not in range(21):
            raise Exception("V4L2 Capture src_id not a number between 0-20")
        self.src_id = src_id
        self.src_str = "/dev/video"+str(int(src_id))
        self.width,self.height = size
        self.fps = fps
        self.open = False
        self.initialized = False
        self.streaming = False
        self.device = -1
        self._open()
        self._init()
        self._start()
        self._buf = None

    def _open(self):
        self.device = dll.open_device(c_char_p(self.src_str))
        self.open = True

    def _init(self):
        if self.open:
            width,height,fps= c_uint32(self.width),c_uint32(self.height),c_uint32(self.fps)
            dll.init_device(self.device,width,height,fps)
            self.initialized = True
            self.width,self.height,self.fps = width.value,height.value,fps.value


    def _start(self):
        if self.initialized:
            dll.start_capturing(self.device)
            self.streaming = True

    def read(self):
        if self._buf:
            buf = self._buf
            dll.release_buffer(self.device,byref(buf))
        buf  = v4l2_buffer()
        buf_ptr =  dll.get_buffer(self.device,byref(buf))
        if buf_ptr:
            buf_ptr = cast(buf_ptr,POINTER(c_uint8*buf.bytesused))
            img = np.frombuffer(buf_ptr.contents,c_uint8)
            img.shape = (self.height,self.width,3)
            timestamp = buf.timestamp.secs+buf.timestamp.usecs/1000000.
            self._buf = buf
            return Frame(timestamp, img)
        else:
            print "Failed to retrieve frame from "+ self.src_str+", Retrying"
            self._buf = None
            return self.read() 

    def read_copy(self):
        buf  = v4l2_buffer()
        buf_ptr =  dll.get_buffer(self.device,byref(buf))
        buf_ptr = cast(buf_ptr,POINTER(c_uint8*buf.bytesused))
        if buf_ptr.contents:
            a = np.frombuffer(buf_ptr.contents,c_uint8)
            a.shape = (self.height,self.width,3)
            b = a.copy()
            dll.release_buffer(self.device,byref(buf))
            return True, b
        else:
            print "Grab error, retrying"
            dll.release_buffer(self.device,byref(buf))
            return self.read() 
    

    def _stop(self):
        if self.streaming:
            dll.stop_capturing(self.device)
            self.streaming = False

    def _uninit(self):
        if self.initialized:
            dll.uninit_device(self.device)
            self.initialized = False


    def _close(self):
        if self.open:
            self.device = dll.close_device(self.device)
            self.open=False
            print "Closed: "+self.src_str

    def __del__(self):
        self._stop()
        self._uninit()
        self._close()

### Debugging
if __name__ == '__main__':
    import numpy as np
    import cv2
    from time import sleep

    # cap = cv2.VideoCapture(0)
    # cap.set(3,1920)
    # cap.set(4,1080)
    # print cap.get(3)
    # for x in range(900):
    #     s,img = cap.read()
    # device =  dll.open_device(c_char_p("/dev/video0"))
    # width,height,fps = c_uint32(1920),c_uint32(1080),c_uint32(30)
    # dll.init_device(device,width,height,fps)
    # dll.start_capturing(device)
    # for x in range(300):
    #     # s= img.copy()
    #     # np.random.random((1920,1080,3))
    #     # dll.mainloop(device)
    #     buf  = v4l2_buffer()
    #     buf_ptr =  dll.get_buffer(device,byref(buf))
    #     buf_ptr = cast(buf_ptr,POINTER(c_uint8*buf.bytesused))
    #     a = np.frombuffer(buf_ptr.contents,c_uint8)
    #     a.shape = (height.value,width.value,3)
    #     # print a.shape
    #     # print buf.index
    #     # sleep(1)
    #     # print a[:9]
    #     # np.save("img.npy",a)
    #     # b = a.copy()
    #     # del a
    #     # del buf_ptr
    #     dll.release_buffer(device,byref(buf))
    # print "stopping"
    # dll.stop_capturing(device)
    # dll.uninit_device(device)
    # dll.close_device(device)
    # # dll.fprintf(stderr, "\n")
    cap = VideoCapture(0,(1280,720),30)
    for x in range(300):
        img = cap.read()
        # print img.shape
    del cap