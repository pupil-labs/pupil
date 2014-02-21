'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

# Python string to/from CFString conversion helper functions:

from ctypes import *
from ctypes import util

cf = cdll.LoadLibrary(util.find_library('CoreFoundation'))

# Setup return types for functions that return pointers.
# (Otherwise ctypes returns 32-bit int which breaks on 64-bit systems.)
# Note that you must also wrap the return value with c_void_p before
# you use it as an argument to another function, otherwise ctypes will
# automatically convert it back to a 32-bit int again.
cf.CFDictionaryCreateMutable.restype = c_void_p
cf.CFStringCreateWithCString.restype = c_void_p
cf.CFAttributedStringCreate.restype = c_void_p
cf.CFDataCreate.restype = c_void_p
cf.CFNumberCreate.restype = c_void_p

# Core Foundation constants
kCFStringEncodingUTF8 = 0x08000100
kCFStringEncodingMacRoman = 0
kCFStringEncodingWindowsLatin1 = 0x0500
kCFStringEncodingISOLatin1 = 0x0201
kCFStringEncodingNextStepLatin = 0x0B01
kCFStringEncodingASCII = 0x0600
kCFStringEncodingUnicode = 0x0100
kCFStringEncodingUTF8 = 0x08000100
kCFStringEncodingNonLossyASCII = 0x0BFF
kCFStringEncodingUTF16 = 0x0100
kCFStringEncodingUTF16BE = 0x10000100
kCFStringEncodingUTF16LE = 0x14000100
kCFStringEncodingUTF32 = 0x0c000100
kCFStringEncodingUTF32BE = 0x18000100
kCFStringEncodingUTF32LE = 0x1c000100
kCFNumberSInt32Type   = 3


def CFSTR(text):
    return c_void_p(cf.CFStringCreateWithCString(None, text.encode('utf8'), kCFStringEncodingASCII))

def cfstring_to_string(cfstring):
    length = cf.CFStringGetLength(cfstring)
    size = cf.CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingASCII)
    buffer = c_buffer(size + 1)
    result = cf.CFStringGetCString(cfstring, buffer, len(buffer), kCFStringEncodingASCII)
    if result:
        return buffer.value

def cfstring_to_string_release(cfstring):
    length = cf.CFStringGetLength(cfstring)
    size = cf.CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingASCII)
    buffer = c_buffer(size + 1)
    result = cf.CFStringGetCString(cfstring, buffer, len(buffer), kCFStringEncodingASCII)
    cf.CFRelease(cfstring)
    if result:
        return buffer.value

def release(cfstring):
    cf.CFRelease(cfstring)


if __name__ == '__main__':
    cf_pointer = CFSTR("THIS is a Test")
    print cfstring_to_string(cf_pointer)
