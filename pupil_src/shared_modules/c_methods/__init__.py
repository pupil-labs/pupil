'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
This file contains bindings and interfaces to functions written in C for speed;
All C source files, a makefile and binaries are in the same folder as this file.
Only wrappers are exposed not the loaded libraries.
"""

from ctypes import *
from numpy.ctypeslib import ndpointer
import os,sys

### Get location of  this file


# source_loc = os.path.dirname(os.path.abspath(__file__))

if getattr(sys, 'frozen', False):
    # we are running in a |PyInstaller| bundle
    dll_path = 'methods.so'
else:
    # we are running in a normal Python environment
    basedir = os.path.dirname(__file__)

    ### Run Autocompiler
    #  Binaries are not distributed instead a make file and source are in the c-methods folder
    #  Make is invoked when this module is imported or run.

    arch_64bit = sizeof(c_void_p) == 8
    if arch_64bit:
        c_flags = "CFLAGS=-m64"
    else:
        c_flags = "CFLAGS=-m32"

    from subprocess import check_output
    # print "c-methods: compiling now."
    compiler_status = check_output(["make",c_flags],cwd=basedir)
    # print "c-methods:",compiler_status
    del check_output
    # print "c-methods: compiling done."
    dll_path = basedir + os.path.sep + 'methods.so'

    ### C-Types binary loading
    if not os.path.isfile(dll_path):
        raise Exception("c-methods Error could not compile binary.")

__methods_dll = CDLL(dll_path)


### C-Types Argtypes and Restype
__methods_dll.filter.argtypes = [ndpointer(c_float),  # integral image
                                c_size_t,           # rows/shape[0]
                                c_size_t,           # cols/shape[1]
                                POINTER(c_int),     # maximal response top left anchor pos height
                                POINTER(c_int),     # maximal response top left anchor pos width
                                POINTER(c_int)]     # maxinal response window size
### C-Types Argtypes and Restype
__methods_dll.ring_filter.argtypes = [ndpointer(c_float),  # integral image
                                    c_size_t,           # rows/shape[0]
                                    c_size_t,           # cols/shape[1]
                                    POINTER(c_int),     # maximal response top left anchor pos height
                                    POINTER(c_int),     # maximal response top left anchor pos width
                                    POINTER(c_int),     # maxinal response window size
                                    POINTER(c_float)]     # maxinal response


### Function Wrappers
def eye_filter(integral):
    rows, cols = integral.shape[0],integral.shape[1]
    x, y, w = c_int(), c_int(), c_int()
    __methods_dll.filter(integral,rows,cols,x,y,w)
    return x.value,y.value,w.value

### Function Wrappers
def ring_filter(integral):
    rows, cols = integral.shape[0],integral.shape[1]
    x, y, w, r = c_int(), c_int(), c_int(), c_float()
    __methods_dll.ring_filter(integral,rows,cols,x,y,w,r)
    return x.value,y.value,w.value,r.value



### Debugging
if __name__ == '__main__':
    import numpy as np
    import cv2
    img = np.ones((1000,1000),dtype=c_uint8)
    # img = np.random.rand((100))
    # img = img.reshape(10,-1)
    # img *=20;
    # img = np.array(img,dtype = c_uint8)
    # img +=20
    img[50:80,100:130] = 0
    # print img
    integral = cv2.integral(img)
    integral =  np.array(integral,dtype=c_float)
    print ring_filter(integral)