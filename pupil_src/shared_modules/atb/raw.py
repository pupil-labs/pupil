#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
import sys,os
import ctypes, ctypes.util
from ctypes import c_int, c_char_p, c_void_p, py_object, c_char
from constants import *


###on linux OpenGL needs to be imported for atb to load
import platform
os_name = platform.system()
del platform
if os_name == "Linux":
    import OpenGL.GL


if getattr(sys, 'frozen', False):
    # we are running in a |PyInstaller| bundle using the local version
    if os_name == "Linux":
        filename = 'libAntTweakBar.so'
    elif os_name == 'Darwin':
        filename = 'libAntTweakBar.dylib'
    else:
        filename = 'libAntTweakBar.dll'
    dll_path = os.path.join(sys._MEIPASS,filename)

else:
    # we are running in a normal Python environment
    dll_path = ctypes.util.find_library('AntTweakBar')

if not dll_path:
    raise RuntimeError, 'AntTweakBar library not found'
__dll__ = ctypes.CDLL(dll_path)

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
TwSetCurrentWindow = __dll__.TwSetCurrentWindow

TwEventMouseButtonGLFW = __dll__.TwEventMouseButtonGLFW
# TwEventMouseMotionGLUT = __dll__.TwEventMousePosGLFW
# TwEventKeyboardGLFW    = __dll__.TwEventKeyGLFW
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
TwDeleteBar.argtypes = c_pointer,
TwRemoveAllVars.argtypes = c_pointer,
TwAddSeparator.restype = c_pointer
TwAddSeparator.argtypes = [c_pointer, c_char_p, c_char_p]
TwAddVarRW.restype   = c_pointer
TwAddVarRW.argtypes  = [c_pointer, c_char_p, c_int, c_void_p, c_void_p]
TwRemoveVar.argtypes = [c_pointer,c_char_p]
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
# TwEventKeyboardGLFW.argtypes = [c_int, c_int]
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



#glfw3 key constants fix

glfw3_keymap = (("GLFW_KEY_ESCAPE"          , 256),
("GLFW_KEY_ENTER"           , 257),
("GLFW_KEY_TAB"             , 258),
("GLFW_KEY_BACKSPACE"       , 259),
("GLFW_KEY_INSERT"          , 260),
("GLFW_KEY_DELETE"          , 261),
("GLFW_KEY_RIGHT"           , 262),
("GLFW_KEY_LEFT"            , 263),
("GLFW_KEY_DOWN"            , 264),
("GLFW_KEY_UP"              , 265),
("GLFW_KEY_PAGE_UP"         , 266),
("GLFW_KEY_PAGE_DOWN"       , 267),
("GLFW_KEY_HOME"            , 268),
("GLFW_KEY_END"             , 269),
("GLFW_KEY_CAPS_LOCK"       , 280),
("GLFW_KEY_SCROLL_LOCK"     , 281),
("GLFW_KEY_NUM_LOCK"        , 282),
# ("GLFW_KEY_PRINT_SCREEN"    , 283),
("GLFW_KEY_PAUSE"           , 284),
("GLFW_KEY_F1"              , 290),
("GLFW_KEY_F2"              , 291),
("GLFW_KEY_F3"              , 292),
("GLFW_KEY_F4"              , 293),
("GLFW_KEY_F5"              , 294),
("GLFW_KEY_F6"              , 295),
("GLFW_KEY_F7"              , 296),
("GLFW_KEY_F8"              , 297),
("GLFW_KEY_F9"              , 298),
("GLFW_KEY_F10"             , 299),
("GLFW_KEY_F11"             , 300),
("GLFW_KEY_F12"             , 301),
("GLFW_KEY_F13"             , 302),
("GLFW_KEY_F14"             , 303),
("GLFW_KEY_F15"             , 304),
("GLFW_KEY_F16"             , 305),
("GLFW_KEY_F17"             , 306),
("GLFW_KEY_F18"             , 307),
("GLFW_KEY_F19"             , 308),
("GLFW_KEY_F20"             , 309),
("GLFW_KEY_F21"             , 310),
("GLFW_KEY_F22"             , 311),
("GLFW_KEY_F23"             , 312),
("GLFW_KEY_F24"             , 313),
("GLFW_KEY_F25"             , 314),
("GLFW_KEY_KP_0"            , 320),
("GLFW_KEY_KP_1"            , 321),
("GLFW_KEY_KP_2"            , 322),
("GLFW_KEY_KP_3"            , 323),
("GLFW_KEY_KP_4"            , 324),
("GLFW_KEY_KP_5"            , 325),
("GLFW_KEY_KP_6"            , 326),
("GLFW_KEY_KP_7"            , 327),
("GLFW_KEY_KP_8"            , 328),
("GLFW_KEY_KP_9"            , 329),
("GLFW_KEY_KP_DECIMAL"      , 330),
("GLFW_KEY_KP_DIVIDE"       , 331),
("GLFW_KEY_KP_MULTIPLY"     , 332),
("GLFW_KEY_KP_SUBTRACT"     , 333),
("GLFW_KEY_KP_ADD"          , 334),
("GLFW_KEY_KP_ENTER"        , 335),
("GLFW_KEY_KP_EQUAL"        , 336),
("GLFW_KEY_LEFT_SHIFT"      , 340),
("GLFW_KEY_LEFT_CONTROL"    , 341),
("GLFW_KEY_LEFT_ALT"        , 342),
("GLFW_KEY_LEFT_SUPER"      , 343),
("GLFW_KEY_RIGHT_SHIFT"     , 344),
("GLFW_KEY_RIGHT_CONTROL"   , 345),
("GLFW_KEY_RIGHT_ALT"       , 346),
("GLFW_KEY_RIGHT_SUPER"     , 347),
("GLFW_KEY_MENU"            , 348))
glfw3_keymap = dict(glfw3_keymap)

glfw2_keymap = (("GLFW_KEY_ESCAPE"          ,256+1),
("GLFW_KEY_F1"           ,256+2),
("GLFW_KEY_F2"           ,256+3),
("GLFW_KEY_F3"           ,256+4),
("GLFW_KEY_F4"           ,256+5),
("GLFW_KEY_F5"           ,256+6),
("GLFW_KEY_F6"           ,256+7),
("GLFW_KEY_F7"           ,256+8),
("GLFW_KEY_F8"           ,256+9),
("GLFW_KEY_F9"           ,256+10),
("GLFW_KEY_F10"          ,256+11),
("GLFW_KEY_F11"          ,256+12),
("GLFW_KEY_F12"          ,256+13),
("GLFW_KEY_F13"          ,256+14),
("GLFW_KEY_F14"          ,256+15),
("GLFW_KEY_F15"          ,256+16),
("GLFW_KEY_F16"          ,256+17),
("GLFW_KEY_F17"          ,256+18),
("GLFW_KEY_F18"          ,256+19),
("GLFW_KEY_F19"          ,256+20),
("GLFW_KEY_F20"          ,256+21),
("GLFW_KEY_F21"          ,256+22),
("GLFW_KEY_F22"          ,256+23),
("GLFW_KEY_F23"          ,256+24),
("GLFW_KEY_F24"          ,256+25),
("GLFW_KEY_F25"          ,256+26),
("GLFW_KEY_UP"           ,256+27),
("GLFW_KEY_DOWN"         ,256+28),
("GLFW_KEY_LEFT"         ,256+29),
("GLFW_KEY_RIGHT"        ,256+30),
("GLFW_KEY_LEFT_SHIFT"       ,256+31),
("GLFW_KEY_RIGHT_SHIFT"       ,256+32),
("GLFW_KEY_LEFT_CONTROL"        ,256+33),
("GLFW_KEY_RIGHT_CONTROL"        ,256+34),
("GLFW_KEY_LEFT_ALT"         ,256+35),
("GLFW_KEY_RIGHT_ALT"         ,256+36),
("GLFW_KEY_TAB"          ,256+37),
("GLFW_KEY_ENTER"        ,256+38),
("GLFW_KEY_BACKSPACE"    ,256+39),
("GLFW_KEY_INSERT"       ,256+40),
("GLFW_KEY_DELETE"          ,256+41),
("GLFW_KEY_PAGE_UP"       ,256+42),
("GLFW_KEY_PAGE_DOWN"     ,256+43),
("GLFW_KEY_HOME"         ,256+44),
("GLFW_KEY_END"          ,256+45),
("GLFW_KEY_KP_0"         ,256+46),
("GLFW_KEY_KP_1"         ,256+47),
("GLFW_KEY_KP_2"         ,256+48),
("GLFW_KEY_KP_3"         ,256+49),
("GLFW_KEY_KP_4"         ,256+50),
("GLFW_KEY_KP_5"         ,256+51),
("GLFW_KEY_KP_6"         ,256+52),
("GLFW_KEY_KP_7"         ,256+53),
("GLFW_KEY_KP_8"         ,256+54),
("GLFW_KEY_KP_9"         ,256+55),
("GLFW_KEY_KP_DIVIDE"    ,256+56),
("GLFW_KEY_KP_MULTIPLY"  ,256+57),
("GLFW_KEY_KP_SUBTRACT"  ,256+58),
("GLFW_KEY_KP_ADD"       ,256+59),
("GLFW_KEY_KP_DECIMAL"   ,256+60),
("GLFW_KEY_KP_EQUAL"     ,256+61),
("GLFW_KEY_KP_ENTER"     ,256+62),
("GLFW_KEY_NUM_LOCK"  ,256+63),
("GLFW_KEY_CAPS_LOCK"    ,256+64),
("GLFW_KEY_SCROLL_LOCK"  ,256+65),
("GLFW_KEY_PAUSE"        ,256+66),
("GLFW_KEY_LEFT_SUPER"       ,256+67),
("GLFW_KEY_RIGHT_SUPER"       ,256+68),
("GLFW_KEY_MENU"         ,256+69))
glfw2_keymap = dict(glfw2_keymap)

mapping = dict([(glfw3_keymap[key],glfw2_keymap[key]) for key in glfw3_keymap.keys()])

def TwEventKeyboardGLFW(key,action):
    try:
        key = mapping[key]
    except KeyError:
        key = key
    __dll__.TwEventKeyGLFW.argtypes = [c_int, c_int]
    return __dll__.TwEventKeyGLFW(key,action)




