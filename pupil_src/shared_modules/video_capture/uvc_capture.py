'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import uvc
from uvc import device_list
#check versions for our own depedencies as they are fast-changing
assert uvc.__version__ >= '0.1'

from ctypes import c_double
from pyglui import ui
from time import time
#logging
import logging
logger = logging.getLogger(__name__)

class CameraCaptureError(Exception):
    """General Exception for this module"""
    def __init__(self, arg):
        super(CameraCaptureError, self).__init__()
        self.arg = arg



class Camera_Capture(object):
    """
    Camera Capture is a class that encapsualtes uvc.Capture:
     - adds UI elements
     - adds timestamping sanitization fns.
    """
    def __init__(self,uid,timebase=None):
        if timebase == None:
            logger.debug("Capture will run with default system timebase")
            self.timebase = c_double(0)
        elif hasattr(timebase,'value'):
            logger.debug("Capture will run with app wide adjustable timebase")
            self.timebase = timebase
        else:
            logger.error("Invalid timebase variable type. Will use default system timebase")
            self.timebase = c_double(0)

        self.use_hw_ts = self.check_hw_ts_support()
        self._last_timestamp = self.get_now()
        self.capture = uvc.Capture(uid)
        self.uid = uid
        if 'C930e' in self.capture.name:
            logger.debug('Timestamp offset for c930 applied: -0.1sec')
            self.ts_offset = -0.1
        else:
            self.ts_offset = 0.0

        if "USB 2.0 Camera" in self.capture.name:
            self.capture.bandwidth_factor = 1.2
            
        logger.debug('avaible modes %s'%self.capture.avaible_modes)

        controls_dict = dict([(c.display_name,c) for c in self.capture.controls])
        try:
            controls_dict['Auto Focus'].value = 0
        except KeyError:
            pass
        try:
            # Auto Exposure Priority = 1 leads to reduced framerates under low light and corrupt timestamps.
            controls_dict['Auto Exposure Priority'].value = 0
        except KeyError:
            pass

        self.sidebar = None
        self.menu = None


    def check_hw_ts_support(self):
        # hw timestamping:
        # uvc supports Sart of Exposure hardware timestamping ofr UVC Capture devices
        # these HW timestamps are excellent referece times and
        # preferred over softwaretimestamp denoting the avaibleilt of frames to the user.
        # however not all uvc cameras report valid hw timestamps, notably microsoft hd-6000
        # becasue all used devices need to properly implement hw timestamping for it to be usefull
        # but we cannot now what device the other process is using  + the user may select a differet capture device during runtime
        # we use some fuzzy logic to determine if hw timestamping should be employed.
        return False
        blacklist = ["Microsoft","HD-6000"]
        qualifying_devices = ["C930e","Integrated Camera", "USB 2.0 Camera"]
        attached_devices = [c.name for c in device_list()]
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

    def re_init(self,uid,size=(640,480),fps=30):

        current_size = self.capture.frame_size
        current_fps = self.capture.frame_rate

        self.capture = None
        #recreate the bar with new values
        menu_conf = self.menu.configuration
        self.deinit_gui()

        self.use_hw_ts = self.check_hw_ts_support()
        self.capture = uvc.Capture(uid)
        self.uid = uid

        self.frame_size = current_size
        self.frame_rate = current_fps
        controls_dict = dict([(c.display_name,c) for c in self.capture.controls])
        try:
            controls_dict['Auto Focus'].value = 0
        except KeyError:
            pass
        # try:
        #     # exposure_auto_priority == 1
        #     # leads to reduced framerates under low light and corrupt timestamps.
        #     self.capture.set_control(controls_dict['Exposure, Auto Priority']['id'], 0)
        # except KeyError:
        #     pass

        self.init_gui(self.sidebar)
        self.menu.configuration = menu_conf

        if 'C930e' in self.capture.name:
            logger.debug('Timestamp offset for c930 applied: -0.1sec')
            self.ts_offset = -0.1
        else:
            self.ts_offset = 0.0


    def get_frame(self):
        try:
            frame = self.capture.get_frame_robust()
        except:
            raise CameraCaptureError("Could not get frame from %s"%self.uid)

        timestamp = frame.timestamp
        if self.use_hw_ts:
            # lets make sure this timestamps is sane:
            if abs(timestamp-uvc.get_sys_time_monotonic()) > 2: #hw_timestamp more than 2secs away from now?
                logger.warning("Hardware timestamp from %s is reported to be %s but monotonic time is %s"%('/dev/video'+str(self.src_id),timestamp,uvc.get_sys_time_monotonic()))
                timestamp = uvc.get_sys_time_monotonic()
        else:
            # timestamp = uvc.get_sys_time_monotonic()
            timestamp = self.get_now()+self.ts_offset

        timestamp -= self.timebase.value
        frame.timestamp = timestamp
        return frame

    def get_now(self):
        return time()

    @property
    def frame_rate(self):
        return self.capture.frame_rate
    @frame_rate.setter
    def frame_rate(self,new_rate):
        #closest match for rate
        rates = [ abs(r-new_rate) for r in self.capture.frame_rates ]
        best_rate_idx = rates.index(min(rates))
        rate = self.capture.frame_rates[best_rate_idx]
        if rate != new_rate:
            logger.warning("%sfps capture mode not available at (%s) on '%s'. Selected %sfps. "%(new_rate,self.capture.frame_size,self.capture.name,rate))
        self.capture.frame_rate = rate


    @property
    def settings(self):
        settings = {}
        settings['name'] = self.capture.name
        settings['frame_rate'] = self.frame_rate
        settings['uvc_controls'] = {}
        for c in self.capture.controls:
            settings['uvc_controls'][c.display_name] = c.value
        return settings
    @settings.setter
    def settings(self,settings):
        try:
            self.frame_rate = settings['frame_rate']
        except KeyError:
            pass

        if settings.get('name','') == self.capture.name:
            for c in self.capture.controls:
                try:
                    c.value = settings['uvc_controls'][c.display_name]
                except KeyError as e:
                    logger.warning('Could not set UVC setting "%s" from last session.'%c.display_name)
    @property
    def frame_size(self):
        return self.capture.frame_size
    @frame_size.setter
    def frame_size(self,new_size):
        self.capture.frame_size = filter_sizes(self.name,new_size)

    @property
    def name(self):
        return self.capture.name

    def init_gui(self,sidebar):

        #lets define some  helper functions:
        def gui_load_defaults():
            for c in self.capture.controls:
                try:
                    c.value = c.def_val
                except:
                    pass

        def gui_update_from_device():
            for c in self.capture.controls:
                c.refresh()

        def gui_init_cam_by_uid(requested_id):
            for cam in uvc.device_list():
                if cam['uid'] == requested_id:
                    self.re_init(requested_id)
                    return
            logger.warning("could not reinit capture, src_id not valid anymore")
            return

        #create the menu entry
        self.menu = ui.Growing_Menu(label='Camera Settings')
        cameras = uvc.device_list()
        camera_names = [c['name'] for c in cameras]
        camera_ids = [c['uid'] for c in cameras]
        self.menu.append(ui.Selector('uid',self,selection=camera_ids,labels=camera_names,label='Capture Device', setter=gui_init_cam_by_uid) )

        # hardware_ts_switch = ui.Switch('use_hw_ts',self,label='use hardware timestamps')
        # hardware_ts_switch.read_only = True
        # self.menu.append(hardware_ts_switch)


        sensor_control = ui.Growing_Menu(label='Sensor Settings')
        sensor_control.collapsed=False
        image_processing = ui.Growing_Menu(label='Image Post Processing')
        image_processing.collapsed=True

        sensor_control.append(ui.Selector('frame_rate',self, selection=self.capture.frame_rates,label='Frames per second' ) )


        for control in self.capture.controls:
            c = None
            ctl_name = control.display_name

            #now we add controls
            if control.d_type == bool :
                c = ui.Switch('value',control,label=ctl_name, on_val=control.max_val, off_val=control.min_val)
            elif control.d_type == int:
                c = ui.Slider('value',control,label=ctl_name,min=control.min_val,max=control.max_val,step=control.step)
            elif type(control.d_type) == dict:
                selection = [value for name,value in control.d_type.iteritems()]
                labels = [name for name,value in control.d_type.iteritems()]
                c = ui.Selector('value',control, label = ctl_name, selection=selection,labels = labels)
            else:
                pass
            # if control['disabled']:
            #     c.read_only = True
            # if ctl_name == 'Exposure, Auto Priority':
            #     # the controll should always be off. we set it to 0 on init (see above)
            #     c.read_only = True

            if c is not None:
                if control.unit == 'processing_unit':
                    image_processing.append(c)
                else:
                    sensor_control.append(c)

        self.menu.append(sensor_control)
        self.menu.append(image_processing)
        self.menu.append(ui.Button("refresh",gui_update_from_device))
        self.menu.append(ui.Button("load defaults",gui_load_defaults))

        self.sidebar = sidebar
        #add below geneal settings
        self.sidebar.insert(1,self.menu)

    def deinit_gui(self):
        if self.menu:
            self.sidebar.remove(self.menu)
            self.menu = None



    def close(self):
        self.deinit_gui()
        # self.capture.close()
        del self.capture
        logger.info("Capture released")


def filter_sizes(cam_name,size):
    #here we can force some defaulit formats
    if "6000" in cam_name:
        if size[0] == 640:
            logger.info("HD-6000 camera selected. Forcing format to 640,360")
            return 640,360
        elif size[0] == 320:
            logger.info("HD-6000 camera selected. Forcing format to 320,360")
            return 320,160
    return size


