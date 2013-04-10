"""
This file contains bindings and interfaces to function writte in C for speed;
All c source files, makefiles and binaries are in the same folder as this file.
Only wrappers are exposed not the loaded libraries.
"""

from ctypes import *
from numpy.ctypeslib import ndpointer

import os.path
dll_name = "methods.so"
dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
__methods_dll = CDLL(dllabspath)


__methods_dll.filter.argtypes = [ndpointer(c_float),  # integral image
                                c_size_t,           # rows/shape[0]
                                c_size_t,           # cols/shape[1]
                                POINTER(c_int),     # maximal response pos height
                                POINTER(c_int),     # maximal response pos width
                                POINTER(c_int)]     # maxinal response window size

def eye_filter(integral):
    rows,cols = integral.shape[0],integral.shape[1]
    x,y,w = c_int(),c_int(),c_int()
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