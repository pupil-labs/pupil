
import cv2
import numpy as np
from methods import normalize,denormalize
from gl_utils import draw_gl_point,draw_gl_point_norm,draw_gl_polyline

import atb
import audio



class Nine_Point_Detector(object):
    """docstring for Nine_Point_"""
    def __init__(self, global_calibrate,shared_pos,screen_marker_pos,screen_marker_state,atb_pos=(0,0)):
        self.active = False
        self.detected = False
        self.global_calibrate = global_calibrate
        self.global_calibrate.value = False
        self.shared_pos = shared_pos
        self.pos = 0,0 # 0,0 is used to indicate no point detected
        self.var1 = c_int(0)

        self.shared_cal9_active = shared_cal9_active
        self.shared_cal9_active.value = True
        self.shared_stage = shared_stage
        self.shared_step = shared_step
        self.shared_circle_id = shared_circle_id
        self.stage = 0
        self.step = 0
        self.next = False
        self.map = (0, 2, 7, 16, 21, 23, 39, 40, 42)
        self.grid_points = None

        atb_label = "9 Point Detector"

      # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = "9_Point_Reference_Detector", label=atb_label,
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 150))
        self._bar.add_button("  begin calibrating  ", self.start)
        self._bar.add_separator("Sep1")
        self._bar.add_var("9 point stage", getter=self.get_stage)
        self._bar.add_var("9 point step", getter=self.get_step)

    def get_stage(self):
        return self.stage

    def get_step(self):
        return self.step

    def start(self):
        audio.say("Starting 9 Point Calibration")
        self.global_calibrate.value = True
        self.shared_pos = 0,0
        self.active = True

    def stop(self):
        audio.say("Stopping Calibration")
        self.global_calibrate.value = False
        self.reset()
        self.publish()
        self.active = False

    def new_ref(self,pos):
        """
        gets called when the user clicks on the world window screen
        """
        pass


    def detect(self,img):
        if self.active:
            # Statemachine
            if self.step > 30:
                self.step = 0
                self.stage += 1
                self.next = False
            # done exit now (is_done() will now return True)
            if self.stage > 8:
                return
            # Detection
            self.pos = 0,0
            self.detected = False

            if self.step in range(10, 25):
                status, self.grid_points = cv2.findCirclesGridDefault(img, (4,11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                if status:
                    self.detected = True
                    img_pos = self.grid_points[self.map[self.stage]][0]
                    self.pos = normalize(img_pos, (img.shape[1],img.shape[0]),flip_y=True)
            # Advance
            if self.next or 1:
                self.step += 1

            self.publish()



    def advance(self):
        self.next=True

    def publish(self):
        self.shared_stage.value = self.stage
        self.shared_step.value = self.step
        self.shared_circle_id.value = self.map[self.stage]
        self.shared_pos = self.pos

    def reset(self):
        self.step = 0
        self.stage = 0
        self.is_done = False
        self.pos = 0,0

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """
        if self.detected:
            draw_gl_polyline(self.grid_points[:,0],(0.,0.,1.,.5), type="Strip")

    def del_bar(self):
        """Delete the ATB bar manually.
            Python's garbage collector doesn't work on the object otherwise
            Due to the fact that ATB is a c library wrapped in ctypes

        """
        self._bar.destroy()
        del self._bar


    def __del__(self):
        self.reset()
        self.publish()
        self.global_calibrate.value = False
