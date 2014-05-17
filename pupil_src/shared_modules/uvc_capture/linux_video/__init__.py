'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs UG

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


from v4l2_capture import VideoCapture,CameraCaptureError
from v4l2_ctl import Controls, Camera_List, Cam
import atb
from ctypes import c_bool
from time import sleep
#logging
import logging
logger = logging.getLogger(__name__)




class Camera_Capture(object):
    """docstring for uvcc_camera"""
    def __init__(self,cam,size=(640,480),fps=None,timebase=None):
        self.src_id = cam.src_id
        self.serial = cam.serial
        self.name = cam.name
        self.controls = Controls(self.src_id)
        try:
            self.controls['focus_auto'].set_val(0)
        except KeyError:
            pass
        try:
            # exposure_auto_priority == 1 
            # leads to reduced framerates under low light and corrupt timestamps.
            self.controls['exposure_auto_priority'].set_val(0)
        except KeyError:
            pass
        self.timebase = timebase
        self.use_hw_ts = self.check_hw_ts_support()

        #give camera some time to change settings.
        sleep(0.3)
        self.capture = VideoCapture(self.src_id,size,fps,timebase = self.timebase, use_hw_timestamps = self.use_hw_ts)
        self.get_frame = self.capture.read
        self.get_now = self.capture.get_time_monotonic



    def check_hw_ts_support(self):
        # hw timestamping:
        # v4l2 supports Sart of Exposure hardware timestamping ofr UVC Capture devices
        # these HW timestamps are excellent referece times and 
        # prefferec over softwaretimestamp denoting the avaibleilt of frames to the user.
        # however not all uvc cameras report valid hw timestamps, notably microsoft hd-6000
        # becasue all used devices need to properly implement hw timestamping for it to be usefull
        # but we cannot now what device the other process is using  + the user may select a differet capture device during runtime
        # we use some fuzzy logic to determine if hw timestamping should be employed.

        blacklist = ["Microsoft","HD-6000"]
        qualifying_devices = ["C930e","Integrated Camera"]
        attached_devices = [c.name for c in Camera_List()]
        if any(qd in self.name for qd in qualifying_devices):
            use_hw_ts = True
            logger.info("Capture device: '%s' supports HW timestamping. Using hardware timestamps." %self.name)
        else:
            use_hw_ts = False
            logger.info("Capture device: '%s' is not known to support HW timestamping. Using software timestamps." %self.name)

        for d in attached_devices:
            if any(bd in d for bd in blacklist):
                logger.info("Capture device: '%s' detected as attached device. Falling back to software timestamps"%d)
                use_hw_ts = False
        return use_hw_ts

    def re_init(self,cam,size=(640,480),fps=30):

        current_size = self.capture.get_size()
        current_fps = self.capture.get_rate()

        self.capture.cleanup()
        self.capture = None
        #recreate the bar with new values
        bar_pos = self.bar._get_position()
        self.bar.destroy()


        self.src_id = cam.src_id
        self.serial = cam.serial
        self.name = cam.name
        self.controls = Controls(self.src_id)

        try:
            self.controls['focus_auto'].set_val(0)
        except KeyError:
            pass
        try:
            self.controls['exposure_auto_priority'].set_val(0)
        except KeyError:
            pass

        self.use_hw_ts = self.check_hw_ts_support()

        self.capture = VideoCapture(self.src_id,current_size,current_fps,self.timebase,self.use_hw_ts)
        self.get_frame = self.capture.read
        self.get_now = self.capture.get_time_monotonic
        self.create_atb_bar(bar_pos)

    def re_init_cam_by_src_id(self,requested_id):
        for cam in Camera_List():
            if cam.src_id == requested_id:
                self.re_init(cam)
                return
        logger.warning("could not reinit capture, src_id not valid anymore")
        return


    def create_atb_bar(self,pos):
        # add uvc camera controls to a separate ATB bar
        size = (200,200)

        self.bar = atb.Bar(name="Camera", label=self.name,
            help="UVC Camera Controls", color=(50,50,50), alpha=100,
            text='light',position=pos,refresh=2., size=size)
        cameras_enum = atb.enum("Capture",dict([(c.name,c.src_id) for c in Camera_List()]) )
        self.bar.add_var("Capture",vtype=cameras_enum,getter=lambda:self.src_id, setter=self.re_init_cam_by_src_id)

        self.bar.add_var('framerate', vtype = atb.enum('framerate',self.capture.rates_menu), getter = lambda:self.capture.current_rate_idx, setter=self.capture.set_rate_idx )
        self.bar.add_var('hardware timestamps',vtype=atb.TW_TYPE_BOOL8,getter=lambda:self.use_hw_ts)

        sorted_controls = [c for c in self.controls.itervalues()]
        sorted_controls.sort(key=lambda c: c.order)


        for control in sorted_controls:
            name = control.atb_name
            if control.type=="bool":
                self.bar.add_var(name,vtype=atb.TW_TYPE_BOOL8,getter=control.get_val,setter=control.set_val)
            elif control.type=='int':
                self.bar.add_var(name,vtype=atb.TW_TYPE_INT32,getter=control.get_val,setter=control.set_val)
                self.bar.define(definition='min='+str(control.min),   varname=name)
                self.bar.define(definition='max='+str(control.max),   varname=name)
                self.bar.define(definition='step='+str(control.step), varname=name)
            elif control.type=="menu":
                if control.menu is None:
                    vtype = None
                else:
                    vtype= atb.enum(name,control.menu)
                self.bar.add_var(name,vtype=vtype,getter=control.get_val,setter=control.set_val)
                if control.menu is None:
                    self.bar.define(definition='min='+str(control.min),   varname=name)
                    self.bar.define(definition='max='+str(control.max),   varname=name)
                    self.bar.define(definition='step='+str(control.step), varname=name)
            else:
                pass
            if control.flags == "inactive":
                pass
            if control.name == 'exposure_auto_priority':
                # the controll should always be off. we set it to 0 on init (see above)
                self.bar.define(definition='readonly=1',varname=control.name)

        self.bar.add_button("refresh",self.controls.update_from_device)
        self.bar.add_button("load defaults",self.controls.load_defaults)

        return size

    def close(self):
        self.kill_atb_bar()
        del self.capture
        logger.info("Capture released")

    def kill_atb_bar(self):
        if hasattr(self,'bar'):
            self.bar.destroy()
            del self.bar
