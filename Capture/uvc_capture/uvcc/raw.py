
from ctypes import *


import os
import sys
# Path = os.path.dirname(os.path.abspath(sys.argv[0]))
# DLL_location = os.path.join(Path,'uvcc.so')
# DLL_location = os.path.join(Path,'c/lsusb.so')

DLL_location = 'uvcc.so'
_uvcc_dll_ = CDLL(DLL_location)


class uvccModelId(Structure):
    _fields_ = [("idVendor", c_uint16 ),
                ("idProduct", c_uint16 )]


class IOUSBDeviceDescriptor(Structure):
        _fields_ = [("bLength", c_uint8),
                    ("bDescriptorType", c_uint8),
                    ("bcdUSB", c_uint16),
                    ("bDeviceClass", c_uint8),
                    ("bDeviceSubClass", c_uint8),
                    ("bDeviceProtocol", c_uint8),
                    ("bMaxPacketSize0", c_uint8),
                    ("idVendor", c_uint16),
                    ("idProduct", c_uint16),
                    ("bcdDevice", c_uint16),
                    ("iManufacturer", c_uint8),
                    ("iProduct", c_uint8),
                    ("iSerialNumber", c_uint8),
                    ("bNumConfigurations", c_uint8)]

class uvccCam(Structure):
    _fields_ = [("idLocation", c_uint32 ),
                ("mId", POINTER(uvccModelId) ),
                ("devDesc", IOUSBDeviceDescriptor),
                ("devIf", POINTER(POINTER(IOUSBDeviceDescriptor)) ), # IOUSBDeviceInterface197
                ("ctrlIf", POINTER(POINTER(IOUSBDeviceDescriptor)) ), #IOUSBDeviceInterface197
                ("ifNo", c_uint8 )]


_uvcc_dll_.uvccGetCamList.argtypes = [POINTER(POINTER(POINTER(uvccCam)))]
_uvcc_dll_.uvccReleaseCamList.argtypes = [POINTER(POINTER(uvccCam)),c_int]


_uvcc_dll_.uvccCamManufacturer.argtypes = [POINTER(uvccCam),c_wchar_p]

if __name__ == '__main__':
    print _uvcc_dll_.uvccInit()
    cam_list = pointer(pointer(uvccCam()))
    cam_n =  _uvcc_dll_.uvccGetCamList(byref(cam_list))
    # print cam_n,cam_list
    # print cam_list.contents.contents.devDesc.iManufacturer

    CamManufacturer = create_unicode_buffer(128)
    _uvcc_dll_.uvccCamManufacturer(cam_list.contents,CamManufacturer)
    str =  CamManufacturer.raw
    print str
    # print str.split("\x00")[0]
    _uvcc_dll_.uvccReleaseCamList(cam_list,cam_n)
    print _uvcc_dll_.uvccExit()

