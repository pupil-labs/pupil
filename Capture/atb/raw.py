#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
import ctypes, ctypes.util
from ctypes import c_int, c_char_p, c_void_p, py_object, c_char
from constants import *

# temp hacky fix, on Ubuntu libAntTweakBar.so
# will only load if loaded through the glumpy module before
# we keep glumpe as a dependency for now anyways
import platform
os_name = platform.system()
del platform
if os_name == "Linux":
    try:
        from glumpy import atb
        del atb
    except:
        pass
# fix end

name = ctypes.util.find_library('AntTweakBar')
if not name:
    raise RuntimeError, 'AntTweakBar library not found'
__dll__ = ctypes.CDLL(name)

TwInit         = __dll__.TwInit
TwTerminate    = __dll__.TwTerminate
TwGetLastError = __dll__.TwGetLastError
TwNewBar       = __dll__.TwNewBar
TwDeleteBar    = __dll__.TwDeleteBar
TwDeleteAllBars= __dll__.TwDeleteAllBars
TwAddSeparator = __dll__.TwAddSeparator
TwAddVarRW     = __dll__.TwAddVarRW
TwAddVarRO     = __dll__.TwAddVarRO
TwAddVarCB     = __dll__.TwAddVarCB
TwAddButton    = __dll__.TwAddButton
TwGetBarName   = __dll__.TwGetBarName
TwGetParam     = __dll__.TwGetParam
TwSetParam     = __dll__.TwSetParam
TwDefineEnum   = __dll__.TwDefineEnum
TwDefine       = __dll__.TwDefine
TwDraw         = __dll__.TwDraw
TwWindowSize   = __dll__.TwWindowSize
TwRemoveAllVars= __dll__.TwRemoveAllVars
TwRemoveVar    = __dll__.TwRemoveVar
TwRefreshBar   = __dll__.TwRefreshBar
TwSetTopBar    = __dll__.TwSetTopBar
TwKeyPressed   = __dll__.TwKeyPressed
TwMouseButton  = __dll__.TwMouseButton
TwMouseMotion  = __dll__.TwMouseMotion
TwWindowSize   = __dll__.TwWindowSize
TwMouseWheel   = __dll__.TwMouseWheel

TwEventMouseButtonGLFW = __dll__.TwEventMouseButtonGLFW
# TwEventMouseMotionGLUT = __dll__.TwEventMousePosGLFW
TwEventKeyboardGLFW    = __dll__.TwEventKeyGLFW
TwEventCharGLFW    = __dll__.TwEventCharGLFW
# TwEventSpecialGLUT     = __dll__.TwEventSpecialGLUT
# TwEventMouseWheelGLUT  = __dll__.TwEventMouseWheelGLFW

#detect 64bit
arch_64bit = ctypes.sizeof(ctypes.c_void_p) == 8
if arch_64bit:
    # On Mac OS Snow Leopard, the following definitions seems to be necessary to
    # ensure 64bits pointers anywhere needed
    c_pointer = ctypes.c_ulonglong
else:
    #normal
    c_pointer = ctypes.c_void_p

TwGetLastError.restype = c_char_p
TwGetBarName.restype = c_char_p
TwNewBar.restype = c_pointer
TwAddSeparator.restype = c_pointer
TwAddSeparator.argtypes = [c_pointer, c_char_p, c_char_p]
TwAddVarRW.restype   = c_pointer
TwAddVarRW.argtypes  = [c_pointer, c_char_p, c_int, c_void_p, c_void_p]
TwAddVarRO.restype   = c_pointer
TwAddVarRO.argtypes  = [c_pointer, c_char_p, c_int, c_void_p, c_void_p]
TwAddVarCB.restype   = c_pointer
TwAddVarCB.argtypes  = [c_pointer, c_char_p, c_int, c_void_p, c_void_p, py_object, c_char_p]
TwAddButton.argtypes = [c_pointer, c_char_p, c_void_p, py_object, c_char_p]
TwRefreshBar.argtypes = [c_pointer]
TwGetParam.restype = c_int
TwGetParam.argtypes =  [c_pointer, c_char_p, c_char_p, c_int, c_int, c_void_p]
TwSetParam.restype = c_int
TwSetParam.argtypes =  [c_pointer, c_char_p, c_char_p, c_int, c_int, c_void_p]
TwEventKeyboardGLFW.argtypes = [c_int, c_int]
TwEventCharGLFW.argtypes = [c_int, c_int]
# TwEventSpecialGLUT.argtypes  = [c_int, c_int, c_int]


# Callback prototypes
BUTTON_FUNC = ctypes.CFUNCTYPE(c_void_p, c_void_p)
SET_FUNC    = ctypes.CFUNCTYPE(c_void_p, c_void_p, c_void_p)
GET_FUNC    = ctypes.CFUNCTYPE(c_void_p, c_void_p, c_void_p)
ERROR_FUNC  = ctypes.CFUNCTYPE(c_void_p, c_char_p)


# struct TwEnumVal
class TwEnumVal(ctypes.Structure):
    _fields_ = [("Value", c_int), ("Label", c_char_p)]

