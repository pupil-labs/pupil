'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import cv2
import numpy as np
from pyglui.cygl.utils import draw_points_norm, draw_polyline, RGBA
from OpenGL.GL import GL_POLYGON
from circle_detector import CircleTracker
from . finish_calibration import finish_calibration

import audio

from pyglui import ui
from . calibration_plugin_base import Calibration_Plugin
# logging
import logging
logger = logging.getLogger(__name__)


class Manual_Marker_Calibration(Calibration_Plugin):
    """
        CircleTracker looks for proper markers
        Using at least 9 positions/points within the FOV
        Ref detector will direct one to good positions with audio cues
        Calibration only collects data at the good positions
    """
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.pos = None
        self.smooth_pos = 0.,0.
        self.smooth_vel = 0.
        self.sample_site = (-2,-2)
        self.counter = 0
        self.counter_max = 30

        self.stop_marker_found = False
        self.auto_stop = 0
        self.auto_stop_max = 30

        self.menu = None

        self.circle_tracker = CircleTracker()
        self.markers = []

    def init_ui(self):
        super().init_ui()
        self.menu.label = "Manual Calibration"
        self.menu.append(ui.Info_Text("Calibrate gaze parameters using a handheld marker."))

    def start(self):
        super().start()
        audio.say("Starting {}".format(self.mode_pretty))
        logger.info("Starting {}".format(self.mode_pretty))
        self.active = True
        self.ref_list = []
        self.pupil_list = []

    def stop(self):
        audio.say("Stopping  {}".format(self.mode_pretty))
        logger.info('Stopping  {}'.format(self.mode_pretty))
        self.screen_marker_state = 0
        self.active = False
        self.smooth_pos = 0.,0.
        # self.close_window()
        self.button.status_text = ''
        if self.mode == 'calibration':
            finish_calibration(self.g_pool, self.pupil_list, self.ref_list)
        elif self.mode == 'accuracy_test':
            self.finish_accuracy_test(self.pupil_list, self.ref_list)
        super().stop()

    def on_notify(self, notification):
        '''
        Reacts to notifications:
           ``calibration.should_start``: Starts the calibration procedure
           ``calibration.should_stop``: Stops the calibration procedure

        Emits notifications:
            ``calibration.started``: Calibration procedure started
            ``calibration.stopped``: Calibration procedure stopped
            ``calibration.marker_found``: Steady marker found
            ``calibration.marker_moved_too_quickly``: Marker moved too quickly
            ``calibration.marker_sample_completed``: Enough data points sampled

        '''
        super().on_notify(notification)

    def recent_events(self, events):
        """
        gets called once every frame.
        reference positon need to be published to shared_pos
        if no reference was found, publish 0,0
        """
        frame = events.get('frame')
        if self.active and frame:
            gray_img = frame.gray

            # Update the marker
            self.markers = self.circle_tracker.update(gray_img)

            self.stop_marker_found = False
            if len(self.markers):
                # Set the pos to be the center of the first detected marker
                marker_pos = self.markers[0]['img_pos']
                self.pos = self.markers[0]['norm_pos']
                # Check if there are stop markers
                for marker in self.markers:
                    if marker['marker_type'] == 'Stop':
                        self.auto_stop += 1
                        self.stop_marker_found = True
                        break
            else:
                self.pos = None  # indicate that no reference is detected

            if self.stop_marker_found is False:
                self.auto_stop = 0

            # Check if there are more than one markers
            if len(self.markers) > 1:
                audio.tink()
                logger.warning("{} markers detected. Please remove all the other markers".format(len(self.markers)))

            # tracking logic
            if len(self.markers) and not self.stop_marker_found:
                # start counter if ref is resting in place and not at last sample site
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
                if self.counter <= 0:
                    if self.smooth_vel < 0.01 and sample_ref_dist > 0.1:
                        self.sample_site = self.smooth_pos
                        audio.beep()
                        logger.debug("Steady marker found. Starting to sample {} datapoints".format(self.counter_max))
                        self.notify_all({'subject':'calibration.marker_found','timestamp':self.g_pool.get_timestamp(),'record':True})
                        self.counter = self.counter_max

                if self.counter > 0:
                    if self.smooth_vel > 0.01:
                        audio.tink()
                        logger.warning("Marker moved too quickly: Aborted sample. Sampled {} datapoints. Looking for steady marker again.".format(self.counter_max-self.counter))
                        self.notify_all({'subject':'calibration.marker_moved_too_quickly','timestamp':self.g_pool.get_timestamp(),'record':True})
                        self.counter = 0
                    else:
                        self.counter -= 1
                        ref = {}
                        ref["norm_pos"] = self.pos
                        ref["screen_pos"] = marker_pos
                        ref["timestamp"] = frame.timestamp
                        self.ref_list.append(ref)
                        if events.get('fixations', []):
                            self.counter -= 5
                        if self.counter <= 0:
                            #last sample before counter done and moving on
                            audio.tink()
                            logger.debug("Sampled {} datapoints. Stopping to sample. Looking for steady marker again.".format(self.counter_max))
                            self.notify_all({'subject':'calibration.marker_sample_completed','timestamp':self.g_pool.get_timestamp(),'record':True})

            # Always save pupil positions
            self.pupil_list.extend(events['pupil_positions'])

            if self.counter:
                if len(self.markers):
                    self.button.status_text = 'Sampling Gaze Data'
                else:
                    self.button.status_text = 'Marker Lost'
            else:
                self.button.status_text = 'Looking for Marker'

            # Stop if autostop condition is satisfied:
            if self.auto_stop >=self.auto_stop_max:
                self.auto_stop = 0
                self.stop()

        else:
            pass


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

        if self.active and len(self.markers):
            # draw the largest ellipse of all detected markers
            for marker in self.markers:
                e = marker['ellipses'][-1]
                pts = cv2.ellipse2Poly( (int(e[0][0]),int(e[0][1])),
                                    (int(e[1][0]/2),int(e[1][1]/2)),
                                    int(e[-1]),0,360,15)
                draw_polyline(pts,color=RGBA(0.,1.,0,1.))
                if len(self.markers) > 1:
                    draw_polyline(pts, 1, RGBA(1., 0., 0., .5), line_type=GL_POLYGON)

            # draw indicator on the first detected marker
            if self.counter and self.markers[0]['marker_type'] == 'Ref':
                e = self.markers[0]['ellipses'][-1]
                pts = cv2.ellipse2Poly( (int(e[0][0]),int(e[0][1])),
                                    (int(e[1][0]/2),int(e[1][1]/2)),
                                    int(e[-1]),0,360,360//self.counter_max)
                indicator = [e[0]] + pts[self.counter:].tolist()[::-1] + [e[0]]
                draw_polyline(indicator, color=RGBA(0.1,.5,.7,.8),line_type=GL_POLYGON)

            # draw indicator on the stop marker(s)
            if self.auto_stop:
                for marker in self.markers:
                    if marker['marker_type'] == 'Stop':
                        e = marker['ellipses'][-1]
                        pts = cv2.ellipse2Poly( (int(e[0][0]),int(e[0][1])),
                                            (int(e[1][0]/2),int(e[1][1]/2)),
                                            int(e[-1]),0,360,360//self.auto_stop_max)
                        indicator = [e[0]] + pts[self.auto_stop:].tolist() + [e[0]]
                        draw_polyline(indicator,color=RGBA(8.,0.1,0.1,.8),line_type=GL_POLYGON)
        else:
            pass

    def deinit_ui(self):
        """gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self.active:
            self.stop()
        super().deinit_ui()
