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

__dll = CDLL(dllabspath)


### C-Types Argtypes and Restype
# __dll.filter.argtypes = [ndpointer(c_float),  # integral image
#                                 c_size_t,           # rows/shape[0]
#                                 c_size_t,           # cols/shape[1]
#                                 POINTER(c_int),     # maximal response top left anchor pos height
#                                 POINTER(c_int),     # maximal response top left anchor pos width
#                                 POINTER(c_int)]     # maxinal response window size

__dll.open_device.argtypes = [c_char_p,]
__dll.open_device.restype = c_void_p
__dll.close_device.argtypes = [c_void_p,]
__dll.close_device.restype = c_int
__dll.get_buffer.restype = c_void_p
__dll.release_buffer.restype = c_int


class timeval(Structure):
    _fields_ = [('tv_sec', c_long),
                ('tv_usec', c_long)]

class v4l2_buffer_mem(Union):
    _fields_ = [("offset",c_uint32),
                ("userptr", c_ulong),
                ("planes", c_void_p), #struct v4l2_planes
                ('fd', c_int32)]

class v4l2_buffer(Structure):
    _fields_ = [('index', c_uint32 ),
                ('type', c_uint32),
                ('bytesused', c_uint32),
                ('flags', c_uint32),
                ('field',c_uint32),
                ('timestamp', timeval),
                ('timecode', c_void_p), # struct timecode
                ('sequence', c_uint32),
                # memory location
                ('memory', c_uint32),
                ('m', v4l2_buffer_mem),
                ('length', c_uint32),
                ('reserved2', c_uint32),
                ('reserved', c_uint32)]


# open device
# setup fmt
# create and map buffers
# qbuffers and start stream
# dque buffer and return POINTER
# qbuffer repead above
# stop stream
# umap buffer

### Debugging
if __name__ == '__main__':
    import numpy as np
    import cv2


    # cap = cv2.VideoCapture(0)
    # cap.set(3,1920)
    # cap.set(4,1080)
    # print cap.get(3)
    # for x in range(900):
    #     s,img = cap.read()
    device =  __dll.open_device(c_char_p("/dev/video0"))
    __dll.init_device(device)
    __dll.start_capturing(device)
    for x in range(900):
        # s= img.copy()
        # np.random.random((1920,1080,3))
        # __dll.mainloop(device)
        buf  = v4l2_buffer()
        buf_ptr =  __dll.get_buffer(device,byref(buf))
        # buf_ptr = cast(buf_ptr,POINTER(c_uint8*buf.bytesused))
        # a = np.frombuffer(buf_ptr.contents,c_uint8)
        # a.shape = (1080,1920,3)
        # print a.shape
        # print a[:9]
        # np.save("img.npy",a)
        # del a
        # del buf_ptr
        __dll.release_buffer(device,byref(buf))
    print "stopping"
    __dll.stop_capturing(device)
    __dll.uninit_device(device)
    print __dll.close_device(device)
    # __dll.fprintf(stderr, "\n")