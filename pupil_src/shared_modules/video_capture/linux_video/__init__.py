'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import v4l2
#check versions for our own depedencies as they are fast-changing
assert v4l2.__version__ >= '0.1'

from ctypes import c_double
from pyglui import ui
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

        self.capture = v4l2.Capture('/dev/video'+str(self.src_id))
        self.capture.frame_size = size
        self.capture.frame_rate = (1,fps or 30)
        self.controls = self.capture.enum_controls()
        controls_dict = dict([(c['name'],c) for c in self.controls])
        try:
            self.capture.set_control(controls_dict['Focus, Auto']['id'], 0)
        except KeyError:
            pass
        try:
            # exposure_auto_priority == 1
            # leads to reduced framerates under low light and corrupt timestamps.
            self.capture.set_control(controls_dict['Exposure, Auto Priority']['id'], 0)
        except KeyError:
            pass

        self.sidebar = None
        self.menu = None


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
        qualifying_devices = ["C930e","Integrated Camera", "USB 2.0 Camera"]
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
        self.deinit_gui()

        self.src_id = cam.src_id
        self.name = cam.name

        self.use_hw_ts = self.check_hw_ts_support()
        self.capture = v4l2.Capture('/dev/video'+str(self.src_id))
        self.capture.frame_size = current_size
        self.capture.frame_rate = (1,current_fps or 30)
        self.controls = self.capture.enum_controls()
        controls_dict = dict([(c['name'],c) for c in self.controls])
        try:
            self.capture.set_control(controls_dict['Focus, Auto']['id'], 0)
        except KeyError:
            pass
        try:
            # exposure_auto_priority == 1
            # leads to reduced framerates under low light and corrupt timestamps.
            self.capture.set_control(controls_dict['Exposure, Auto Priority']['id'], 0)
        except KeyError:
            pass

        self.init_gui(self.sidebar)



    def get_frame(self):
        try:
            frame = self.capture.get_frame_robust()
        except:
            raise CameraCaptureError("Could not get frame from %s"%self.src_id)

        timestamp = frame.timestamp
        if self.use_hw_ts:
            # lets make sure this timestamps is sane:
            if abs(timestamp-v4l2.get_sys_time_monotonic()) > 2: #hw_timestamp more than 2secs away from now?
                logger.warning("Hardware timestamp from %s is reported to be %s but monotonic time is %s"%('/dev/video'+str(self.src_id),timestamp,v4l2.get_sys_time_monotonic()))
                timestamp = v4l2.get_sys_time_monotonic()
        else:
            timestamp = v4l2.get_sys_time_monotonic()

        timestamp -= self.timebase.value
        frame.timestamp = timestamp
        return frame

    def get_now(self):
        return v4l2.get_sys_time_monotonic()

    @property
    def frame_rate(self):
        #return rate as denominator only
        return float(self.capture.frame_rate[1])/self.capture.frame_rate[0]
    @frame_rate.setter
    def frame_rate(self, rate):
        if isinstance(rate,(tuple,list)):
            self.capture.frame_rate = rate
        elif isinstance(rate,(int,float)):
            self.capture.frame_rate = 1. , rate
        else:
            raise Exception("Please set rate as '(num,den)' or as 'den' assuming num is 1")

    @property
    def frame_size(self):
        return self.capture.frame_size
    @frame_size.setter
    def frame_size(self, value):
        self.capture.frame_size = value



    def init_gui(self,sidebar):

        #lets define some  helper functions:
        def gui_load_defaults():
            for c in self.controls:
                if not c['disabled']:
                    self.capture.set_control(c['id'],c['default'])
                    c['value'] = self.capture.get_control(c['id'])

        def gui_update_from_device():
            for c in self.controls:
                if not c['disabled']:
                    c['value'] = self.capture.get_control(c['id'])


        def gui_get_frame_rate():
            return self.capture.frame_rate

        def gui_set_frame_rate(rate):
            self.capture.frame_rate = rate

        def gui_init_cam_by_src_id(requested_id):
            for cam in Camera_List():
                if cam.src_id == requested_id:
                    self.re_init(cam)
                    return
            logger.warning("could not reinit capture, src_id not valid anymore")
            return

        #create the menu entry
        self.menu = ui.Growing_Menu(label='Camera Settings')
        cameras = Camera_List()
        camera_names = [c.name for c in cameras]
        camera_ids = [c.src_id for c in cameras]
        self.menu.append(ui.Selector('src_id',self,selection=camera_ids,labels=camera_names,label='Capture Device', setter=gui_init_cam_by_src_id) )

        hardware_ts_switch = ui.Switch('use_hw_ts',self,label='use hardware timestamps')
        hardware_ts_switch.read_only = True
        self.menu.append(hardware_ts_switch)

        self.menu.append(ui.Selector('frame_rate', selection=self.capture.frame_rates,labels=[str(d/float(n)) for n,d in self.capture.frame_rates],
                                        label='Frame Rate', getter=gui_get_frame_rate, setter=gui_set_frame_rate) )


        for control in self.controls:
            c = None
            ctl_name = control['name']

            # we use closures as setters and getters for each control element
            def make_setter(control):
                def fn(val):
                    self.capture.set_control(control['id'],val)
                    control['value'] = self.capture.get_control(control['id'])
                return fn
            def make_getter(control):
                def fn():
                    return control['value']
                return fn
            set_ctl = make_setter(control)
            get_ctl = make_getter(control)

            #now we add controls
            if control['type']=='bool':
                c = ui.Switch(ctl_name,getter=get_ctl,setter=set_ctl)
            elif control['type']=='int':
                c = ui.Slider(ctl_name,getter=get_ctl,min=control['min'],max=control['max'],
                                step=control['step'], setter=set_ctl)

            elif control['type']=="menu":
                if control['menu'] is None:
                    selection = range(control['min'],control['max']+1,control['step'])
                    labels = selection
                else:
                    selection = [value for name,value in control['menu'].iteritems()]
                    labels = [name for name,value in control['menu'].iteritems()]
                c = ui.Selector(ctl_name,getter=get_ctl,selection=selection,labels = labels,setter=set_ctl)
            else:
                pass
            if control['disabled']:
                c.read_only = True
            if ctl_name == 'Exposure, Auto Priority':
                # the controll should always be off. we set it to 0 on init (see above)
                c.read_only = True

            if c is not None:
                self.menu.append(c)

        self.menu.append(ui.Button("refresh",gui_update_from_device))
        self.menu.append(ui.Button("load defaults",gui_load_defaults))
        self.menu.collapsed = True
        self.sidebar = sidebar
        #add below geneal settings
        self.sidebar.insert(1,self.menu)

    def deinit_gui(self):
        if self.menu:
            self.sidebar.remove(self.menu)
            self.menu = None



    def close(self):
        self.deinit_gui()
        self.capture.close()
        del self.capture
        logger.info("Capture released")



