from v4l2_capture import VideoCapture
from v4l2_ctl import Controls, Camera_List, Cam



class Camera_Capture(object):
    """docstring for uvcc_camera"""
    def __init__(self,cam,size=(640,480),fps=None):
        self.src_id = cam.src_id
        self.serial = cam.serial
        self.name = cam.name
        self.controls = Controls(self.src_id)
        self.capture = VideoCapture(self.src_id,size,fps)

        self.get_frame = self.capture.read


    def set_size(self,size):
        pass

    def get_size(self):
        return self.capture.width,self.capture.height

    def set_fps(self,fps):
        pass

    def get_fps(self):
        return self.capture.fps

    def make_atb_bar(self,pos):
        return pos

    def kill_atb_bar(self,pos):
        return pos