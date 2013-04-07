
from ctypes import *


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



###control requests
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
                '__UVCC_REQ_OUT_OF_RANGE')

control_dict = dict(zip(uvcc_controls,range(len(uvcc_controls))))

import os.path
dll_name = "uvcc.so"
dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
__uvcc_dll = CDLL(dllabspath)

__uvcc_dll.uvccInit.argtypes = []
__uvcc_dll.uvccExit.argtypes = []
__uvcc_dll.uvccGetCamList.argtypes = [POINTER(POINTER(POINTER(uvccCam)))]
__uvcc_dll.uvccReleaseCamList.argtypes = [POINTER(POINTER(uvccCam)),c_int]
__uvcc_dll.uvccGetCamWithQTUniqueID.argtypes = [c_char_p,POINTER(uvccCam)]
__uvcc_dll.uvccGetCamWithModelId.argtypes = [POINTER(uvccModelId),POINTER(uvccCam)]
__uvcc_dll.uvccReleaseCam.argtypes = [POINTER(uvccCam)]
__uvcc_dll.uvccCamQTUniqueID.argtypes = [POINTER(uvccCam),c_wchar_p] # 19
__uvcc_dll.uvccCamQTUniqueID.restypes = [c_char_p]
__uvcc_dll.uvccCamManufacturer.argtypes = [POINTER(uvccCam),c_wchar_p] #128
__uvcc_dll.uvccCamProduct.argtypes = [POINTER(uvccCam),c_wchar_p] #128
__uvcc_dll.uvccCamSerialNumber.argtypes = [POINTER(uvccCam),c_wchar_p] #128
__uvcc_dll.uvccUni2Char.argtypes = [c_wchar_p,c_char_p,c_int,c_int] #128
__uvcc_dll.uvccSendRequest.argtypes = [POINTER(uvccCam),c_uint8, c_uint,c_void_p]
__uvcc_dll.uvccSendRequest.restypes = [c_uint]
__uvcc_dll.uvccRequestInfo.argtypes = [POINTER(uvccCam), c_uint]
__uvcc_dll.uvccRequestInfo.restypes = [c_int8]

def uvccInit():
    __uvcc_dll.uvccInit()

def uvccExit():
    __uvcc_dll.uvccExit()

def uvccGetCamList(cam_list):
    return __uvcc_dll.uvccGetCamList(cam_list)

def uvccReleaseCamList(cam_list,cam_n):
    return __uvcc_dll.uvccReleaseCamList(cam_list,cam_n)

def uvccSendRequest(control,request,camera):
    bRequest = request
    val = c_int(0)
    err = __uvcc_dll.uvccSendRequest(camera,bRequest,control_dict[control],byref(val))
    if err == 0:
        return val.value

def uvccCamProduct(camera):
    uni_buf = create_unicode_buffer(128)
    uni_buf_len = __uvcc_dll.uvccCamProduct(camera,uni_buf)
    if uni_buf_len < 0:
        print "Error: could not get Camera Name"
        return None
    else:
        str_buf = create_string_buffer(uni_buf_len)
        str_len  = __uvcc_dll.uvccUni2Char(uni_buf,str_buf,uni_buf_len,0)
        return str_buf.value

def uvccCamManufacturer(camera):
    uni_buf = create_unicode_buffer(128)
    uni_buf_len = __uvcc_dll.uvccCamManufacturer(camera,uni_buf)
    if uni_buf_len < 0:
        print "Error: could not get Camera Manufacturer"
        return None
    else:
        str_buf = create_string_buffer(uni_buf_len)
        str_len  = __uvcc_dll.uvccUni2Char(uni_buf,str_buf,uni_buf_len,0)
        return str_buf.value

def uvccCamSerialNumber(camera):
    uni_buf = create_unicode_buffer(128)
    uni_buf_len = __uvcc_dll.uvccCamSerialNumber(camera,uni_buf)
    if uni_buf_len < 0:
        print "Error: could not get Camera Serial Number"
        return None
    else:
        str_buf = create_string_buffer(uni_buf_len)
        str_len  = __uvcc_dll.uvccUni2Char(uni_buf,str_buf,uni_buf_len,0)
        return str_buf.value


def uvccCamQTUniqueID(camera):
    uni_buf = create_unicode_buffer(22)
    uni_buf_len = __uvcc_dll.uvccCamQTUniqueID(camera,uni_buf)
    if uni_buf_len is None:
        print "Error: could not get Unique QT ID Name"
        return None
    else:
        str_buf = create_string_buffer(22)
        str_len  = __uvcc_dll.uvccUni2Char(uni_buf,str_buf,22,0)
        return str_buf

def uvccGetCamWithModelId(mId):
    cam = uvccCam()
    ret = __uvcc_dll.uvccGetCamWithModelId(mId,cam)
    print ret
    return cam

### simple wrappers
def uvccSetVal(val,control,camera):
    bRequest = UVC_SET_CUR
    val = c_int(val)
    err = __uvcc_dll.uvccSendRequest(camera,bRequest,control_dict[control],byref(val))
    if err == 0:
        return val.value

def uvccGetVal(control,camera):
    bRequest = UVC_GET_CUR
    val = c_int(0)
    err = __uvcc_dll.uvccSendRequest(camera,bRequest,control_dict[control],byref(val))
    if err == 0:
        return val.value


def uvccRequestInfo(control,camera):
    """
    D0 1 = Supports GET value requests      Capability
    D1 1 = Supports SET value requests      Capability
    D2 1 = Disabled due to automatic mode (under device control)    State
    D3 1 = Autoupdate Control   Capability
    D4 1 = Asynchronous Control Capability
    D5 1 = Disabled due to incompatibility with Commit state.   State
    """
    return __uvcc_dll.uvccRequestInfo(camera,control_dict[control])



if __name__ == '__main__':


    __uvcc_dll.uvccInit()
    cam_list = pointer(pointer(uvccCam()))
    cam_n =  __uvcc_dll.uvccGetCamList(cam_list)

    for i in range(cam_n):  # it seems cameras are sorted from high to low for opencv
        print "idVendor",hex(cam_list[i].contents.devDesc.idVendor)
        print "idProduct",hex(cam_list[i].contents.devDesc.idProduct)
        print "ifNo", cam_list[i].contents.idLocation
        print uvccCamProduct(cam_list[i].contents)
        uid = uvccModelId()
        uid = cam_list[i].contents.mId.contents
        cam = cam_list[i].contents
        print 'INFO', hex(__uvcc_dll.uvccRequestInfo(cam,control_dict['UVCC_REQ_FOCUS_ABS']))
        # print get_CamManufacturer(cam_list[i].contents)
        # print get_CamSerialNumber(cam_list[i].contents)

    # print uvccSendRequest("UVCC_REQ_BRIGHTNESS_ABS",UVC_GET_DEF,cam_list[cam_n-1])
    # print uvccGetVal("UVCC_REQ_BRIGHTNESS_ABS",cam_list[cam_n-1])
    # print set_val(0,"UVCC_REQ_BRIGHTNESS_ABS",cam_list[cam_n-1])

    __uvcc_dll.uvccReleaseCamList(cam_list,cam_n)

    __uvcc_dll.uvccExit()

