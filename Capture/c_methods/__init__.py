"""
This file contains bindings and interfaces to function writte in C for speed;
All c source files, makefiles and binaries are in the same folder as this file.
Only wrappers are exposed not the loaded libraries.
"""

from ctypes import *
from numpy.ctypeslib import ndpointer
import os

source_loc = os.path.dirname(os.path.abspath(__file__))

###Run Autocomiler
#  Binaries are not distributed instead a make file and source are in the c-methods folder
#  Make is invoked when this module is imported or run.
# print "c-methods: compiling now."
arch_64bit = sizeof(c_void_p) == 8
if arch_64bit:
    c_flags = "CFLAGS=-m64"
else:
    c_flags = "CFLAGS=-m32"
from subprocess import check_output
compiler_status = check_output(["make",c_flags],cwd=source_loc)
# print "c-methods:",compiler_status
del check_output
# print "c-methods: compiling done."

dll_name = "methods.so"
dllabspath = source_loc + os.path.sep + dll_name
if not os.path.isfile(dllabspath):
    raise Exception("c-methods Error could not find binary.")

__methods_dll = CDLL(dllabspath)

__methods_dll.filter.argtypes = [ndpointer(c_float),  # integral image
                                c_size_t,           # rows/shape[0]
                                c_size_t,           # cols/shape[1]
                                POINTER(c_int),     # maximal response top left anchor pos height
                                POINTER(c_int),     # maximal response top left anchor pos width
                                POINTER(c_int)]     # maxinal response window size

def eye_filter(integral):
    rows, cols = integral.shape[0],integral.shape[1]
    x, y, w = c_int(), c_int(), c_int()
    __methods_dll.filter(integral,rows,cols,x,y,w)
    return x.value,y.value,w.value


if __name__ == '__main__':
    import numpy as np
    import cv2
    img = np.ones((10,10),dtype=c_uint8)
    # img = np.random.rand((100))
    # img = img.reshape(10,-1)
    # img *=20;
    # img = np.array(img,dtype = c_uint8)
    # img +=20
    img[5:6,5:6] = 0
    # print img
    integral = cv2.integral(img)
    integral =  np.array(integral,dtype=c_float)
    print eye_filter(integral)