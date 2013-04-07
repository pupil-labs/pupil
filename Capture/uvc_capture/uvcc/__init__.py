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
        self.atb_name = name[9:].capitalize() #pretify the name
        self.order = i
        self.value = None
        self.assess_type()


    def assess_type(self):
        """
        find out if a control is active
        find out if the range is bool or int
        """
        self.value = None
        self.min = None
        self.max = None
        self.step    = None
        self.default = None
        self.menu    = None #some day in the future we will extract the control menu entries here.

        self.info = self.get_info()
        """
        D0 1 = Supports GET value requests      Capability
        D1 1 = Supports SET value requests      Capability
        D2 1 = Disabled due to automatic mode (under device control)    State
        D3 1 = Autoupdate Control   Capability
        D4 1 = Asynchronous Control Capability
        D5 1 = Disabled due to incompatibility with Commit state.   Statex
        """

        if self.info > 0 :  # Control supported
            self.value = self.get_val_from_device()
            self.min =  self.get_(UVC_GET_MIN)
            self.max =  self.get_(UVC_GET_MAX)
            self.step    =  self.get_(UVC_GET_RES)
            self.default =  self.get_(UVC_GET_DEF)

            if (self.max,self.min) == (None,None):
                self.type  = "bool"
            # elif (self.max,self.min) == (None,None):
            #     ###I guess this should be a menu
            #     self.type  = "int"
            #     self.flags = "active"
            #     self.min = 0
            #     self.max = 20
            #     self.step = 1
            else:
                self.type  = "int"

            if self.info >> 3 & 1: # Disabled due to automatic mode (under device control)
                self.flags = "inactive"
            else:
                self.flags = "active"
        else:
            self.type  = "unknown type"
            self.flags = "control not supported"
            self.value = None

    def get_val_from_device(self):
        return uvccGetVal(self.name,self.handle)

    def get_val(self):
        return self.value

    def set_val(self,val):
        self.value = val
        return uvccSetVal(val,self.name,self.handle)

    def get_info(self):
        return uvccRequestInfo(self.name,self.handle)

    def get_(self,request):
        return uvccSendRequest(self.name,request,self.handle)


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
        controls_str = uvcc_controls

        self.controls = {}
        i = 0
        for c in controls_str:
            self.controls[c] = Control(c,i,self.handle)
            i +=1

    def update_from_device(self):
        for c in self.controls.itervalues():
            if c.flags == "active":
                c.value = c.get_val_from_device()

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
        sort_cams = [self.cam_list[i] for i in range(self.cam_n)]
        sort_cams.sort(key=lambda l:-l.contents.idLocation) #from my tests so far OTKit sorts them by idLocation order

        for i in range(self.cam_n):
            self.append(Camera(sort_cams[i],i))

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
        print cam.cv_id
        cam.init_controls()
        cam.load_defaults()
        for c in cam.controls.itervalues():
            if c.flags != "control not supported":
                print c.name, " "*(40-len(c.name)), c.value, c.min,c.max,c.step
    uvc_cameras.release()