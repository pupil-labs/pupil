from v4l2_capture import VideoCapture
from v4l2_ctl import Controls, Camera_List, Cam
import atb
from time import sleep
#logging
import logging
logger = logging.getLogger(__name__)


class Camera_Capture(object):
    """docstring for uvcc_camera"""
    def __init__(self,cam,size=(640,480),fps=None):
        self.src_id = cam.src_id
        self.serial = cam.serial
        self.name = cam.name
        self.controls = Controls(self.src_id)
        try:
            self.controls['focus_auto'].set_val(0)
        except KeyError:
            pass

        #give camera some time to change settings.
        sleep(0.3)
        self.capture = VideoCapture(self.src_id,size,fps)
        self.get_frame = self.capture.read



    def re_init(self,cam,size=(640,480),fps=30):

        current_size = self.capture.get_size()
        current_fps = self.capture.get_rate()
        self.capture = None

        self.src_id = cam.src_id
        self.serial = cam.serial
        self.name = cam.name
        self.controls = Controls(self.src_id)

        try:
            self.controls['focus_auto'].set_val(0)
        except KeyError:
            pass

        #give camera some time to change settings.
        sleep(0.3)
        self.capture = VideoCapture(self.src_id,size,fps)
        self.get_frame = self.capture.read


        #recreate the bar with new values
        bar_pos = self.bar._get_position()
        self.bar.destroy()
        self.create_atb_bar(bar_pos)

    def re_init_cam_by_src_id(self,src_id):
        try:
            cam = Camera_List()[src_id]
        except KeyError:
            logger.warning("could not reinit capture, src_id not valid anymore")
            return
        self.re_init(cam,self.get_size())


    def create_atb_bar(self,pos):
        # add uvc camera controls to a separate ATB bar
        size = (200,200)

        self.bar = atb.Bar(name="Camera", label=self.name,
            help="UVC Camera Controls", color=(50,50,50), alpha=100,
            text='light',position=pos,refresh=2., size=size)

        cameras_enum = atb.enum("Capture",dict([(c.name,c.src_id) for c in Camera_List()]) )
        self.bar.add_var("Capture",vtype=cameras_enum,getter=lambda:self.src_id, setter=self.re_init_cam_by_src_id)

        self.bar.add_var('framerate', vtype = atb.enum('framerate',self.capture.rates_menu), getter = lambda:self.capture.current_rate_idx, setter=self.capture.set_rate_idx )

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
                # self.bar.define(definition='readonly=1',varname=control.name)

        self.bar.add_button("refresh",self.controls.update_from_device)
        self.bar.add_button("load defaults",self.controls.load_defaults)

        return size

    def close(self):
        self.kill_atb_bar()
        del self.capture

    def kill_atb_bar(self):
        self.bar.destroy()
        del self.bar
