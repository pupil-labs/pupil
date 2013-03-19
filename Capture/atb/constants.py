#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
''' AntTweakBar constants '''

# typedef enum ETwGraphAPI
(TW_OPENGL,
 TW_DIRECT3D9,
 TW_DIRECT3D10) = range(1,4)

# typedef enum ETwMouseButtonID
(TW_MOUSE_LEFT,
 TW_MOUSE_MIDDLE,
 TW_MOUSE_RIGHT) = range(1,4)

# typedef enum ETwMouseAction
(TW_MOUSE_RELEASED,
 TW_MOUSE_PRESSED) = range(0,2)

# typedef enum ETwType
(TW_TYPE_UNDEF,   TW_TYPE_BOOLCPP,  TW_TYPE_BOOL8,   TW_TYPE_BOOL16,
 TW_TYPE_BOOL32,  TW_TYPE_CHAR,     TW_TYPE_INT8,    TW_TYPE_UINT8, 
 TW_TYPE_INT16,   TW_TYPE_UINT16,   TW_TYPE_INT32,   TW_TYPE_UINT32,
 TW_TYPE_FLOAT,   TW_TYPE_DOUBLE,   TW_TYPE_COLOR32, TW_TYPE_COLOR3F,
 TW_TYPE_COLOR4F, TW_TYPE_CDSTRING, _,               TW_TYPE_QUAT4F,
 TW_TYPE_QUAT4D,  TW_TYPE_DIR3F,    TW_TYPE_DIR3D) = range(0,23)

# typedef enum ETwParamValueType
(TW_PARAM_INT32,
 TW_PARAM_FLOAT,
 TW_PARAM_DOUBLE,
 TW_PARAM_CSTRING) = range(0,4)

# typedef enum ETwKeyModifier
TW_KMOD_NONE  = 0x0000  
TW_KMOD_SHIFT = 0x0003
TW_KMOD_CTRL  = 0x00c0
TW_KMOD_ALT   = 0x0100
TW_KMOD_META  = 0x0c00
