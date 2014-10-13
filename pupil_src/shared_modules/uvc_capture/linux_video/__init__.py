'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs UG

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


import v4l2
import atb
from ctypes import c_bool
from time import sleep
#logging
import logging
logger = logging.getLogger(__name__)

class CameraCaptureError(Exception):
    """General Exception for this module"""
    def __init__(self, arg):
        super(CameraCaptureError, self).__init__()
        self.arg = arg

def Camera_List():
    '''
    Thin wrapper around list_devices to improve formatting.
    '''
    class Cam(object):
        pass

    cam_list = []
    for c in v4l2.list_devices():
        cam = Cam()
        cam.name = c['dev_name']
        cam.src_id = int(c['dev_path'][-1])
        cam.bus_info = c['bus_info']
        cam_list.append(cam)
    return cam_list


class Camera_Capture(object):
    """
    Camera Capture is a class that encapsualtes v4l2.Capture: 
     - adds UI elements 
     - adds timestamping sanitization fns.
    """
    def __init__(self,cam,size=(640,480),fps=None,timebase=None):
        self.src_id = cam.src_id
        self.name = cam.name

        self.timebase = timebase
        self.use_hw_ts = self.check_hw_ts_support()
        self._last_timestamp = v4l2.get_sys_time_monotonic()

        self.capture = v4l2.Capture('/dev/video'+str(self.src_id))
        self.capture.frame_size = size
        self.capture.frame_rate = (1,fps or 30)
        self.controls = self.capture.enum_controls()
        self.controls_dict = dict([(c['name'],c) for c in self.controls])
        self._frame_rates = self.capture.frame_rates
        self._atb_frame_rates_dict = dict( [(str(r),idx) for idx,r in enumerate(self._frame_rates)] ) 
        try:
            self.capture.set_control(self.controls_dict['Focus, Auto']['id'], 0)
        except KeyError:
            pass
        try:
            # exposure_auto_priority == 1
            # leads to reduced framerates under low light and corrupt timestamps.
            self.capture.set_control(self.controls_dict['Exposure, Auto Priority']['id'], 0)
        except KeyError:
            pass


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

        current_size = self.capture.frame_size
        current_fps = self.capture.frame_rate[-1]

        self.capture.close()
        self.capture = None
        #recreate the bar with new values
        bar_pos = self.bar._get_position()
        self.bar.destroy()


        self.src_id = cam.src_id
        self.name = cam.name


        self.use_hw_ts = self.check_hw_ts_support()
        self.capture = v4l2.Capture('/dev/video'+str(self.src_id))
        self.capture.frame_size = current_size
        self.capture.frame_rate = (1,current_fps or 30)
        self.controls = self.capture.enum_controls()
        self.controls_dict = dict([(c['name'],c) for c in self.controls])
        try:
            self.capture.set_control(self.controls_dict['Focus, Auto']['id'], 0)
        except KeyError:
            pass
        try:
            # exposure_auto_priority == 1
            # leads to reduced framerates under low light and corrupt timestamps.
            self.capture.set_control(self.controls_dict['Exposure, Auto Priority']['id'], 0)
        except KeyError:
            pass

        self.create_atb_bar(bar_pos)

    def re_init_cam_by_src_id(self,requested_id):
        for cam in Camera_List():
            if cam.src_id == requested_id:
                self.re_init(cam)
                return
        logger.warning("could not reinit capture, src_id not valid anymore")
        return


    def get_frame(self): 
        try:
            frame = self.capture.get_frame_robust()
        except:
            raise CameraCaptureError("Could not get frame from %s"%self.src_id)

        timestamp = frame.timestamp
        if self.use_hw_ts:
                    
            # lets make sure this timestamps is sane:
            if abs(timestamp-v4l2.get_sys_time_monotonic()) > 5: #hw_timestamp more than 5secs away from now?
                logger.warning("Hardware timestamp from %s is reported to be %s but monotonic time is %s and last timestamp was %s"%('/dev/video'+str(self.src_id),timestamp,v4l2.get_sys_time_monotonic(),self._last_timestamp))
                timestamp = self._last_timestamp + self.capture.framerate[0]/float(self.capture.framerate[1])
                logger.warning("Correcting timestamp by extrapolation from last known timestamp using set rate: %s. TS now at %s"%(str(self.capture.framerate),timestamp) )
                self._last_timestamp = timestamp

        else:
            timestamp = v4l2.get_sys_time_monotonic()

            timestamp -= self.timebase.value
        frame.timestamp = timestamp
        return frame

    def atb_set_ctl(self,val,ctl):
        ctl_id = self.controls_dict[ctl]['id']
        self.capture.set_control(ctl_id,val)
        val = self.capture.get_control(ctl_id)
        self.controls_dict[ctl]['value'] = val

    def atb_get_ctl(self,ctl):
        # ctl_id = self.controls_dict[ctl]['id']
        # return self.capture.get_control(ctl_id)
        return self.controls_dict[ctl]['value']

    def atb_load_defaults(self):
        for c in self.controls:
            if not c['disabled']:
                self.capture.set_control(c['id'],c['default'])
                c['value'] = self.capture.get_control(c['id'])


    def atb_get_frame_rate(self):
        return self.capture.frame_rates.index(self.capture.frame_rate)

    def atb_set_frame_rate(self,rate_idx):
        rate = self.capture.frame_rates[rate_idx]
        print rate,rate_idx
        self.capture.frame_rate = rate


    def create_atb_bar(self,pos):
        # add uvc camera controls to a separate ATB bar
        size = (200,200)

        self.bar = atb.Bar(name="Camera", label=self.name,
            help="UVC Camera Controls", color=(50,50,50), alpha=100,
            text='light',position=pos,refresh=2., size=size)
        cameras_enum = atb.enum("Capture",dict([(c.name,c.src_id) for c in Camera_List()]) )
        self.bar.add_var("Capture",vtype=cameras_enum,getter=lambda:self.src_id, setter=self.re_init_cam_by_src_id)

        self.bar.add_var('framerate', vtype = atb.enum('framerate',self._atb_frame_rates_dict), getter = self.atb_get_frame_rate, setter=self.atb_set_frame_rate )
        self.bar.add_var('hardware timestamps',vtype=atb.TW_TYPE_BOOL8,getter=lambda:self.use_hw_ts)

        u_name = 0
        for control in self.controls:
            name = str(u_name)
            u_name +=1
            label =  control['name']
            if control['type']=="bool":
                self.bar.add_var(name,vtype=atb.TW_TYPE_BOOL8,getter=self.atb_get_ctl,setter=self.atb_set_ctl,label = label, data=label)
            elif control['type']=='int':
                self.bar.add_var(name,vtype=atb.TW_TYPE_INT32,getter=self.atb_get_ctl,setter=self.atb_set_ctl,label = label, data=label)
                self.bar.define(definition='min='+str(control['min']),   varname=name)
                self.bar.define(definition='max='+str(control['max']),   varname=name)
                self.bar.define(definition='step='+str(control['step']), varname=name)
            elif control['type']=="menu":
                if control['menu'] == {}:
                    vtype = None
                else:
                    vtype= atb.enum(name,control['menu'])
                self.bar.add_var(name,vtype=vtype,getter=self.atb_get_ctl,setter=self.atb_set_ctl,label = label, data=label)
                if control['menu'] == {}:
                    self.bar.define(definition='min='+str(control['min']),   varname=name)
                    self.bar.define(definition='max='+str(control['max']),   varname=name)
                    self.bar.define(definition='step='+str(control['step']), varname=name)
            else:
                pass
            if control['disabled']:
                self.bar.define(definition='readonly=1',varname=name)
            if label == 'Exposure, Auto Priority':
                # the controll should always be off. we set it to 0 on init (see above)
                self.bar.define(definition='readonly=1',varname=name)

        # self.bar.add_button("refresh",self.controls.update_from_device)
        self.bar.add_button("load defaults",self.atb_load_defaults)

        return size

    def close(self):
        self.kill_atb_bar()
        self.capture.close()
        del self.capture
        logger.info("Capture released")

    def kill_atb_bar(self):
        if hasattr(self,'bar'):
            self.bar.destroy()
            del self.bar

