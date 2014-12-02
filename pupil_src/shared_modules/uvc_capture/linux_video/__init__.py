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
        self.deinit_gui()

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

        self.init_gui(self.sidebar)

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
            pass
            # lets make sure this timestamps is sane:
            # if abs(timestamp-v4l2.get_sys_time_monotonic()) > 5: #hw_timestamp more than 5secs away from now?
            #     logger.warning("Hardware timestamp from %s is reported to be %s but monotonic time is %s and last timestamp was %s"%('/dev/video'+str(self.src_id),timestamp,v4l2.get_sys_time_monotonic(),self._last_timestamp))
            #     timestamp = self._last_timestamp + self.capture.framerate[0]/float(self.capture.framerate[1])
            #     logger.warning("Correcting timestamp by extrapolation from last known timestamp using set rate: %s. TS now at %s"%(str(self.capture.framerate),timestamp) )
            #     self._last_timestamp = timestamp

        else:
            timestamp = v4l2.get_sys_time_monotonic()

        timestamp -= self.timebase.value
        frame.timestamp = timestamp
        return frame

    @property
    def frame_rate(self):
        return self.capture.frame_rate

    property
    def frame_size(self):
        return self.capture.frame_size
    @frame_size.setter
    def frame_size(self, value):
        self.capture.frame_size = value



    def gui_load_defaults(self):
        for c in self.controls:
            if not c['disabled']:
                self.capture.set_control(c['id'],c['default'])
                c['value'] = self.capture.get_control(c['id'])


    # def atb_get_frame_rate(self):
    #     return self.capture.frame_rates.index(self.capture.frame_rate)

    # def atb_set_frame_rate(self,rate_idx):
    #     rate = self.capture.frame_rates[rate_idx]
    #     self.capture.frame_rate = rate



    def init_gui(self,sidebar):


        sorted_controls = [c for c in self.controls.itervalues()]
        sorted_controls.sort(key=lambda c: c.order)

        self.menu = ui.Growing_Menu(label='Camera Settings')

        cameras = Camera_List()
        camera_names = [c.name for c in cameras]
        camera_ids = [c.src_id for c in cameras]
        self.menu.append(ui.Selector('src_id',self,selection=camera_ids,labels=camera_names,label='Capture Device', setter=self.re_init_cam_by_src_id) )

        hardware_ts_switch = ui.Switch('use_hw_ts',self,label='use hardware timestamps')
        hardware_ts_switch.read_only=True
        self.menu.append(hardware_ts_switch)


        for control in sorted_controls:
            c = None
            ctl_name = control['name']

            # we use closures as setters for each control element
            def make_closure(ctl_id):
                def fn(self,val):
                    self.capture.set_control(ctl_id,val)
                    self.controls_dict[ctl]['value'] = self.capture.get_control(ctl_id)
                return fn
            set_ctl = make_closure(control['id'])

            if control['type']=='bool':
                c = ui.Switch(ctl_name,self.controls_dict,setter=set_ctl)
            elif control['type']=='int':
                c = ui.Slider(ctl_name,self.controls_dict,min=control['min'],max=control['max'],
                                step=control['step'], setter=set_ctl)

            elif control['type']=="menu":
                if control.menu is None:
                    selection = range(control['min'],control['max']+1,control['step'])
                    labels = selection
                else:
                    selection = [value for name,value in control.menu.iteritems()]
                    labels = [name for name,value in control.menu.iteritems()]
                c = ui.Selector(ctl_name,self.controls_dict,selection=selection,labels = labels,setter=set_ctl)
            else:
                pass
            if control['disabled']:
                c.read_only = True
            if ctl_name == 'Exposure, Auto Priority':
                # the controll should always be off. we set it to 0 on init (see above)
                c.read_only = True

            if c is not None:
                self.menu.append(c)

        self.menu.append(ui.Button("refresh",self.controls.update_from_device))
        self.menu.append(ui.Button("load defaults",self.gui_load_defaults))
        self.sidebar = sidebar
        self.sidebar.append(self.menu)

    def deinit_gui(self):
        if self.menu:
            self.sidebar.remove(self.menu)
            self.menu = None



    def close(self):
        self.deinit_gui()
        self.capture.close()
        del self.capture
        logger.info("Capture released")



