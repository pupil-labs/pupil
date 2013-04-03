"""
OOP style interface for uvcc c_types binding

Three classes:
    Camera_List intitializes and manages the UVCC library and device_list handle
    Camera holds each device handle, names, controls ect.
    Control is the actual Control with methods for getting and setting them.
"""

from raw import *

class Control(object):
    """docstring for uvcc_Control"""
    def __init__(self,name,i,handle):
        self.handle = handle
        self.name = name
        self.order = i
        self.assess_type()

    def get_val(self):
        return uvccGetVal(self.name,self.handle)

    def set_val(self,val):
        return uvccSetVal(val,self.name,self.handle)

    def get_(self,request):
        return uvccSendRequest(self.name,request,self.handle)

    def assess_type(self):
        """
        find out if a control is active
        find out if the range is bool or int
        """
        self.current = self.get_val()
        self.min = None
        self.max = None
        self.step    = None
        self.default = None
        self.menu    = None #some day in the future we will extract the control menu entries here.

        if self.current != None:
            self.min =  self.get_(UVC_GET_MIN)
            self.max =  self.get_(UVC_GET_MAX)
            self.step    =  self.get_(UVC_GET_RES)
            self.default =  self.get_(UVC_GET_DEF)

            if (self.max,self.min) == (None,None):
                self.type  = "bool"
                self.flags = "active"
            else:
                self.type  = "int"
                self.flags = "active"
        else:
            self.type  = "unknown control type"
            self.flags = "inactive"



class Camera(object):
    """docstring for uvcc_camera"""
    def __init__(self, handle,cv_id):
        self.handle = handle
        self.cv_id = cv_id
        self.name = uvccCamProduct(self.handle)
        self.manufacurer = uvccCamManufacturer(self.handle)
        self.serial = uvccCamSerialNumber(self.handle)
        self.controls = None

    def init_controls(self):
         ###list of all controls implemented by uvcc, the names evaluate to ints using a dict lookup in raw.py
        controls_str=('UVCC_REQ_SCANNING_MODE',
                        'UVCC_REQ_EXPOSURE_AUTOMODE',
                        'UVCC_REQ_EXPOSURE_AUTOPRIO',
                        'UVCC_REQ_EXPOSURE_ABS',
                        'UVCC_REQ_EXPOSURE_REL',
                        'UVCC_REQ_FOCUS_AUTO',
                        'UVCC_REQ_FOCUS_ABS',
                        'UVCC_REQ_FOCUS_REL',
                        'UVCC_REQ_IRIS_ABS',
                        'UVCC_REQ_IRIS_REL',
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
                        ) #'__UVCC_REQ_OUT_OF_RANGE'

        self.controls = {}
        i = 0
        for c in controls_str:
            self.controls[c] = Control(c,i,self.handle)
            i +=1

    def load_defaults(self):
        for c in self.controls.itervalues():
            if c.flags == "active" and c.default is not None:
                c.set_val(c.default)

class Camera_List(list):
    """docstring for uvcc_control"""

    def __init__(self):
        uvccInit()
        self.cam_list = pointer(pointer(uvccCam()))
        self.cam_n = uvccGetCamList(self.cam_list)
        #sort them as the cameras appear in OpenCV VideoCapture
        for i in range(self.cam_n)[::-1]:
            self.append(Camera(self.cam_list[i],i))


    def release(self):
        """
        call when done with class instance
        """
        uvccReleaseCamList(self.cam_list,self.cam_n)
        uvccExit()


if __name__ == '__main__':
    uvc_cameras = Camera_List()
    for cam in uvc_cameras:
        print cam.name
        cam.init_controls()
        cam.load_defaults()
        for c in cam.controls.itervalues():
            if c.flags == "active":
                print c.name, " "*(40-len(c.name)), c.current,c.type, c.min,c.max,c.step
    uvc_cameras.release()