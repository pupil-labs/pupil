from raw import *

class uvcc_camera(object):
    """docstring for uvcc_camera"""
    def __init__(self, handle,cv_id):
        self.handle = handle
        self.cv_id = cv_id
        self.name = uvccCamProduct(self.handle)
        self.manufacurer = uvccCamManufacturer(self.handle)
        self.serial = uvccCamSerialNumber(self.handle)

        self.controls =('UVCC_REQ_SCANNING_MODE',
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
                        '__UVCC_REQ_OUT_OF_RANGE')

    def get_val(self, c):
        return uvccGet_val(c,self.handle)

    def set_val(self, c):
        return uvccSet_val(c,self.handle)

    def get_(self,c,request):
        return uvccGet_(c,request,self.handle)

class uvcc_control(object):
    """docstring for uvcc_control"""

    def __init__(self):
        uvccInit()
        self.cam_list = pointer(pointer(uvccCam()))
        self.cam_n = uvccGetCamList(self.cam_list)
        self.cameras = []
        #sort them as the cameras appear in OpenCV VideoCapture
        for i in range(self.cam_n)[::-1]:
            self.cameras.append(uvcc_camera(self.cam_list[i],i))

    def __getitem__(self,key):
        if key >= self.cam_n:
            raise KeyError("UVCC: Wrong Camera index")
        else:
            return self.cameras[key]


    def terminate(self):
        """
        call when done with class instance
        """
        uvccReleaseCamList(self.cam_list,self.cam_n)
        uvccExit()


if __name__ == '__main__':
    uvc_cameras = uvcc_control()
    print uvc_cameras[1].get_val("UVCC_REQ_FOCUS_AUTO")
    uvc_cameras.terminate()