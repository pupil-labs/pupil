'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
from methods import normalize,denormalize
from file_methods import load_object
from pyglui.cygl.utils import draw_points_norm,draw_polyline,RGBA
from OpenGL.GL import GL_POLYGON
from circle_detector import find_concetric_circles
from . finish_calibration import finish_calibration
from . import calibrate

import audio

from pyglui import ui
from . calibration_plugin_base import Calibration_Plugin

#logging
import logging
logger = logging.getLogger(__name__)

class Adjust_Calibration(Calibration_Plugin):
    """
    """
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.active = False
        self.detected = False
        self.pos = None
        self.smooth_pos = 0.,0.
        self.smooth_vel = 0.
        self.sample_site = (-2,-2)
        self.counter = 0
        self.counter_max = 30
        self.markers = []
        self.world_size = None

        self.stop_marker_found = False
        self.auto_stop = 0
        self.auto_stop_max = 30

        self.menu = None
        self.button = None


    def init_gui(self):

        self.info = ui.Info_Text("Touch up gaze mapping parameters using a single hand held marker.")
        self.g_pool.calibration_menu.append(self.info)

        self.menu = ui.Growing_Menu('Controls')
        self.g_pool.calibration_menu.append(self.menu)

        self.button = ui.Thumb('active',self,label='C',setter=self.toggle,hotkey='c')
        self.button.on_color[:] = (.3,.2,1.,.9)
        self.g_pool.quickbar.insert(0,self.button)

    def deinit_gui(self):
        if self.menu:
            self.g_pool.calibration_menu.remove(self.menu)
            self.g_pool.calibration_menu.remove(self.info)
            self.menu = None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None


    def toggle(self,_=None):
        if self.active:
            self.stop()
        else:
            self.start()

    def start(self):
        logger.info("Starting Touchup")
        self.active = True
        self.ref_list = []
        self.gaze_list = []


    def stop(self):
        logger.info("Stopping Touchup")
        self.smooth_pos = 0.,0.
        self.sample_site = -2,-2
        self.counter = 0
        self.active = False
        self.button.status_text = ''


        offset_pt_clound = calibrate.preprocess_2d_data_monocular(calibrate.closest_matches_monocular(self.ref_list,self.gaze_list) )
        if len(offset_pt_clound)<3:
            logger.error('Did not sample enough data for touchup please retry.')
            return

        #Calulate the offset for gaze to target
        offset_pt_clound = np.array(offset_pt_clound)
        offset =  offset_pt_clound[:,:2]-offset_pt_clound[:,2:]
        mean_offset  = np.mean(offset,axis=0)

        user_calibration = load_object(os.path.join(self.g_pool.user_dir, "user_calibration_data"))

        self.pupil_list = user_calibration['pupil_list']
        self.ref_list = user_calibration['ref_list']
        calibration_method = user_calibration['calibration_method']

        if '3d' in calibration_method:
            logger.error('adjust calibration is not supported for 3d calibration.')
            return

        for r in self.ref_list:
            r['norm_pos'] = [ r['norm_pos'][0]-mean_offset[0],r['norm_pos'][1]-mean_offset[1] ]


        finish_calibration(self.g_pool,self.pupil_list,self.ref_list)


    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        """
        gets called once every frame.
        reference positon need to be published to shared_pos
        if no reference was found, publish 0,0
        """
        if self.active:
            recent_pupil_positions = events['pupil_positions']

            gray_img  = frame.gray

            if self.world_size is None:
                self.world_size = frame.width,frame.height

            self.markers = find_concetric_circles(gray_img,min_ring_count=3)

            if len(self.markers) > 0:
                self.detected = True
                marker_pos = self.markers[0][0][0] #first marker, innermost ellipse, center
                self.pos = normalize(marker_pos,(frame.width,frame.height),flip_y=True)



            else:
                self.detected = False
                self.pos = None #indicate that no reference is detected



            #tracking logic
            if self.detected:
                self.auto_stop +=1
                self.stop_marker_found = True

                # calculate smoothed manhattan velocity
                smoother = 0.3
                smooth_pos = np.array(self.smooth_pos)
                pos = np.array(self.pos)
                new_smooth_pos = smooth_pos + smoother*(pos-smooth_pos)
                smooth_vel_vec = new_smooth_pos - smooth_pos
                smooth_pos = new_smooth_pos
                self.smooth_pos = list(smooth_pos)
                #manhattan distance for velocity
                new_vel = abs(smooth_vel_vec[0])+abs(smooth_vel_vec[1])
                self.smooth_vel = self.smooth_vel + smoother*(new_vel-self.smooth_vel)

                #distance to last sampled site
                sample_ref_dist = smooth_pos-np.array(self.sample_site)
                sample_ref_dist = abs(sample_ref_dist[0])+abs(sample_ref_dist[1])

                # start counter if ref is resting in place and not at last sample site
                if not self.counter:

                    if self.smooth_vel < 0.01 and sample_ref_dist > 0.1:
                        self.sample_site = self.smooth_pos
                        audio.beep()
                        logger.debug("Steady marker found. Starting to sample {} datapoints".format(self.counter_max))
                        self.counter = self.counter_max

                if self.counter:
                    if self.smooth_vel > 0.01:
                        audio.tink()
                        logger.warning("Marker moved to quickly: Aborted sample. Sampled {} datapoints. Looking for steady marker again.".format((self.counter_max-self.counter)))
                        self.counter = 0
                    else:
                        self.counter -= 1
                        ref = {}
                        ref["norm_pos"] = self.pos
                        ref["screen_pos"] = denormalize(self.pos,(frame.width,frame.height),flip_y=True)
                        ref["timestamp"] = frame.timestamp
                        self.ref_list.append(ref)
                        if self.counter == 0:
                            #last sample before counter done and moving on
                            audio.tink()
                            logger.debug("Sampled {} datapoints. Stopping to sample. Looking for steady marker again.".format(self.counter_max))

            #always save pupil positions
            for pt in events.get('gaze_positions',[]):
                if pt['confidence'] > self.pupil_confidence_threshold:
                    #we add an id for the calibration preprocess data to work as is usually expects pupil data.
                    pt['id'] = 0
                    self.gaze_list.append(pt)

            if self.counter:
                if self.detected:
                    self.button.status_text = 'Sampling Gaze Data'
                else:
                    self.button.status_text = 'Marker Lost'
            else:
                self.button.status_text = 'Looking for Marker'

            # stop if autostop condition is satisfied:
            if self.auto_stop >= self.auto_stop_max:
                self.auto_stop = 0
                self.stop()
        else:
            pass

    def get_init_dict(self):
        return {}

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """

        if self.active:
            draw_points_norm([self.smooth_pos],size=15,color=RGBA(1.,1.,0.,.5))

        if self.active and self.detected:
            for marker in self.markers:
                e = marker[-1]
                pts = cv2.ellipse2Poly( (int(e[0][0]),int(e[0][1])),
                                    (int(e[1][0]/2),int(e[1][1]/2)),
                                    int(e[-1]),0,360,15)
                draw_polyline(pts,color=RGBA(0.,1.,0,1.))
        else:
            pass

    def cleanup(self):
        """gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self.active:
            self.stop()
        self.deinit_gui()
