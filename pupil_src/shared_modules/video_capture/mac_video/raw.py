'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from ctypes import *
from cf_string import CFSTR, cfstring_to_string_release
import os,sys

#logging
import logging
logger = logging.getLogger(__name__)


if getattr(sys, 'frozen', False):
    # we are running in a |PyInstaller| bundle
    dll_path = os.path.join(sys._MEIPASS,'uvcc.so')
else:
    ### Get location of  this file
    source_loc = os.path.dirname(os.path.abspath(__file__))
    ### Run Autocompiler
    #  Binaries are not distributed instead a make file and source are in this folder
    #  Make is invoked when this module is imported or run.

    from subprocess import check_output
    logger.debug("Compiling now.")
    compiler_status = check_output(["make"],cwd=source_loc)
    logger.debug('Compiler status: %s'%compiler_status)
    del check_output
    logger.debug("Compiling done.")
    dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uvcc.so')


### defines and constants
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
                ("devIf", c_void_p ), # IOUSBDeviceInterface197
                ("ctrlIf", c_void_p), #IOUSBDeviceInterface197
                ("ifNo", c_uint8 )]



### control requests
UVC_RC_UNDEFINED = 0x00
UVC_SET_CUR      = 0x01
UVC_GET_CUR      = 0x81
UVC_GET_MIN      = 0x82
UVC_GET_MAX      = 0x83
UVC_GET_RES      = 0x84
UVC_GET_LEN      = 0x85
UVC_GET_INFO     = 0x86
UVC_GET_DEF      = 0x87

### we use a dict to mimic an enum
uvcc_controls = ('UVCC_REQ_SCANNING_MODE',
                'UVCC_REQ_EXPOSURE_AUTOMODE',
                'UVCC_REQ_EXPOSURE_AUTOPRIO',
                'UVCC_REQ_EXPOSURE_ABS',
                'UVCC_REQ_EXPOSURE_REL',
                'UVCC_REQ_FOCUS_AUTO',
                'UVCC_REQ_FOCUS_ABS',
                'UVCC_REQ_FOCUS_REL',
                'UVCC_REQ_IRIS_ABS',
                'UVCC_REQ_IRIS_REL',
                'UVCC_REQ_ZOOM_ABS',
                'UVCC_REQ_BACKLIGHT_COMPENSATION_ABS',
                'UVCC_REQ_BRIGHTNESS_ABS',
                'UVCC_REQ_CONTRAST_ABS',
                'UVCC_REQ_GAIN_ABS',
                'UVCC_REQ_POWER_LINE_FREQ',
                'UVCC_REQ_SATURATION_ABS',
                'UVCC_REQ_SHARPNESS_ABS',
                'UVCC_REQ_GAMMA_ABS',
                'UVCC_REQ_WB_TEMP_AUTO',
                'UVCC_REQ_WB_TEMP_ABS',
                'UVCC_REQ_WB_COMPONENT_AUTO',
                'UVCC_REQ_WB_COMPONENT_ABS',
                '__UVCC_REQ_OUT_OF_RANGE')
control_dict = dict(zip(uvcc_controls,range(len(uvcc_controls))))



### import CDLL

__uvcc_dll = CDLL(dll_path)


### return and arg type defs
__CFStringRef = c_void_p

__uvcc_dll.uvccInit.argtypes = []
__uvcc_dll.uvccExit.argtypes = []

__uvcc_dll.uvccGetCamList.argtypes = [POINTER(POINTER(POINTER(uvccCam)))]
__uvcc_dll.uvccGetCamsWithModelID.argtypes = [POINTER(uvccModelId),POINTER(POINTER(POINTER(uvccCam)))]
__uvcc_dll.uvccReleaseCamList.argtypes = [POINTER(POINTER(uvccCam)),c_int]
__uvcc_dll.uvccGetCamWithQTUniqueID.argtypes = [__CFStringRef,POINTER(POINTER(uvccCam))] #__CFStringRef
__uvcc_dll.uvccReleaseCam.argtypes = [POINTER(uvccCam)]

__uvcc_dll.uvccCamQTUniqueID.argtypes = [POINTER(uvccCam)]
__uvcc_dll.uvccCamQTUniqueID.restype = __CFStringRef
__uvcc_dll.uvccCamManufacturer.argtypes = [POINTER(uvccCam)]
__uvcc_dll.uvccCamManufacturer.restype = __CFStringRef
__uvcc_dll.uvccCamProduct.argtypes = [POINTER(uvccCam)]
__uvcc_dll.uvccCamProduct.restype = __CFStringRef
__uvcc_dll.uvccCamSerialNumber.argtypes = [POINTER(uvccCam)]
__uvcc_dll.uvccCamSerialNumber.restype = __CFStringRef

__uvcc_dll.uvccOpenCam.argtypes = [POINTER(uvccCam)]
__uvcc_dll.uvccCloseCam.argtypes = [POINTER(uvccCam)]

__uvcc_dll.uvccUni2Char.argtypes = [c_wchar_p,c_char_p,c_int,c_int]
__uvcc_dll.uvccSendRequest.argtypes = [POINTER(uvccCam),c_uint8, c_uint,c_void_p]
__uvcc_dll.uvccSendInfoRequest.argtypes = [POINTER(uvccCam), c_uint,c_void_p] #uvccCam *cam, enum uvccRequest uvccReq, int8_t *pData


### fn wrappers
def uvccInit():
    __uvcc_dll.uvccInit()

def uvccExit():
    __uvcc_dll.uvccExit()

def uvccGetCamList():
    cam_list = pointer(pointer(uvccCam()))
    cam_n = __uvcc_dll.uvccGetCamList(cam_list)
    return cam_n, cam_list

def uvccReleaseCamList(cam_list,cam_n):
    return __uvcc_dll.uvccReleaseCamList(cam_list,cam_n)

def uvccReleaseCam(cam):
    __uvcc_dll.uvccReleaseCam(cam)

def uvccOpenCam(cam):
    if __uvcc_dll.uvccOpenCam(cam) !=0:
        logger.error("Cam could not be opended")
        return False
    else:
        return True

def uvccCloseCam(cam):
    if __uvcc_dll.uvccCloseCam(cam) !=0:
        logger.error("Cam could not be closed")
        return False
    else:
        return True

def uvccSendRequest(control,request,camera):
    bRequest = request
    val = c_int(0)
    err = __uvcc_dll.uvccSendRequest(camera,bRequest,control_dict[control],byref(val))
    if err == 0:
        return val.value

def uvccRequestInfo(control,camera):
    """
    bitfield
    D0 1 = Supports GET value requests      Capability
    D1 1 = Supports SET value requests      Capability
    D2 1 = Disabled due to automatic mode (under device control)    State
    D3 1 = Autoupdate Control   Capability
    D4 1 = Asynchronous Control Capability
    D5 1 = Disabled due to incompatibility with Commit state.   State
    """
    val = c_int(0)
    err = __uvcc_dll.uvccSendInfoRequest(camera,control_dict[control],byref(val))
    if err == 0:
        return val.value


def uvccCamProduct(camera):
    cf_string = c_void_p(__uvcc_dll.uvccCamProduct(camera))
    if cf_string:
        return cfstring_to_string_release(cf_string)

def uvccCamManufacturer(camera):
    cf_string = c_void_p(__uvcc_dll.uvccCamManufacturer(camera))
    if cf_string:
        return cfstring_to_string_release(cf_string)

def uvccCamSerialNumber(camera):
    cf_string = c_void_p(__uvcc_dll.uvccCamSerialNumber(camera))
    if cf_string:
        return cfstring_to_string_release(cf_string)

def uvccCamQTUniqueID(camera):
    cf_string = c_void_p(__uvcc_dll.uvccCamQTUniqueID(camera))
    if cf_string:
        return cfstring_to_string_release(cf_string)

def uvccGetCamsWithModelID(mId):
    cam_list = pointer(pointer(uvccCam()))
    cam_n = __uvcc_dll.uvccGetCamsWithModelID(mId,cam_list)
    if cam_n > 0 :
        return cam_list,cam_n
    else:
        logger.error("could not add camera that matched uvccModelId: %s"%mId)
        return None,0

def uvccGetCamWithQTUniqueID(uId):
    cf_uId = CFSTR(uId)
    cam = pointer(uvccCam())
    ret = __uvcc_dll.uvccGetCamWithQTUniqueID(cf_uId,cam) #__CFStringRef
    if ret !=0:
        return None
    else:
        return cam

### convinence wrappers
def uvccSetVal(val,control,camera):
    bRequest = UVC_SET_CUR
    val = c_int(val)
    err = __uvcc_dll.uvccSendRequest(camera,bRequest,control_dict[control],byref(val))
    return err

def uvccGetVal(control,camera):
    bRequest = UVC_GET_CUR
    val = c_int(0)
    err = __uvcc_dll.uvccSendRequest(camera,bRequest,control_dict[control],byref(val))
    if err == 0:
        return val.value
    else:
        return None



if __name__ == '__main__':
    uvccInit()
    cam_n,cam_list = uvccGetCamList()
    print "detected cameras:",cam_n
    for i in range(cam_n):
        print "idVendor",hex(cam_list[i].contents.devDesc.idVendor)
        print "idProduct",hex(cam_list[i].contents.devDesc.idProduct)
        print "Location", cam_list[i].contents.idLocation
        print "Product Name:",uvccCamProduct(cam_list[i].contents)
        print "Product Serial:",uvccCamSerialNumber(cam_list[i].contents)
        print "Manufacturer:", uvccCamManufacturer(cam_list[i].contents)
        print "uId:",uvccCamQTUniqueID(cam_list[i].contents)
        uid = uvccCamQTUniqueID(cam_list[i].contents)
        # manually construct uId: (it looks similar to this: 0x1a11000005ac8510)
        # uid = "0x%08x%04x%04x" %(cam_list[i].contents.idLocation,cam_list[i].contents.mId.contents.idVendor,cam_list[i].contents.mId.contents.idProduct)

    # print uvccSendRequest("UVCC_REQ_BRIGHTNESS_ABS",UVC_GET_DEF,cam_list[cam_n-1])
    # print uvccGetVal("UVCC_REQ_BRIGHTNESS_ABS",cam_list[cam_n-1])
    # print set_val(0,"UVCC_REQ_BRIGHTNESS_ABS",cam_list[cam_n-1])
    __uvcc_dll.uvccReleaseCamList(cam_list,cam_n)
    cam = uvccGetCamWithQTUniqueID(uid)
    # # cam = uvccGetCamsWithModelID(mid)
    if cam:
        uvccOpenCam(cam)
        print "Location", cam.contents.idLocation
        print "Product Name:",uvccCamProduct(cam)
        print uvccRequestInfo("UVCC_REQ_EXPOSURE_ABS",cam)
        # val =  uvccGetVal("UVCC_REQ_BRIGHTNESS_ABS",cam)
        # print uvccSetVal(val-1,"UVCC_REQ_BRIGHTNESS_ABS",cam)
        uvccCloseCam(cam)
        uvccReleaseCam(cam)
    __uvcc_dll.uvccExit()

