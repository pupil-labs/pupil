'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
from gl_utils import adjust_gl_view,clear_gl_screen,basic_gl_setup
import OpenGL.GL as gl
from glfw import *
from circle_detector import CircleTracker
from platform import system

import audio

from pyglui import ui
from pyglui.cygl.utils import draw_points, draw_polyline, RGBA
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
from . calibration_plugin_base import Calibration_Plugin
from . finish_calibration import finish_calibration

#logging
import logging
logger = logging.getLogger(__name__)

# window calbacks
def on_resize(window,w,h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)

# easing functions for animation of the marker fade in/out
def easeInOutQuad(t, b, c, d):
    """Robert Penner easing function examples at: http://gizma.com/easing/
    t = current time in frames or whatever unit
    b = beginning/start value
    c = change in value
    d = duration

    """
    t /= d/2
    if t < 1:
        return c/2*t*t + b
    t-=1
    return -c/2 * (t*(t-2) - 1) + b

def interp_fn(t,b,c,d,start_sample=15.,stop_sample=55.):
    # ease in, sample, ease out
    if t < start_sample:
        return easeInOutQuad(t,b,c,start_sample)
    elif t > stop_sample:
        return 1-easeInOutQuad(t-stop_sample,b,c,d-stop_sample)
    else:
        return 1.0


class Screen_Marker_Calibration(Calibration_Plugin):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites - not between

    """
    def __init__(self, g_pool, fullscreen=True, marker_scale=1.0,
                 sample_duration=40, monitor_idx=0):
        super().__init__(g_pool)
        self.screen_marker_state = 0.
        self.sample_duration = sample_duration  # number of frames to sample per site
        self.lead_in = 25  # frames of marker shown before starting to sample
        self.lead_out = 5  # frames of markers shown after sampling is donw

        self.active_site = None
        self.sites = []
        self.display_pos = -1., -1.
        self.on_position = False
        self.pos = None

        self.marker_scale = marker_scale

        self._window = None

        self.menu = None

        self.monitor_idx = monitor_idx
        self.fullscreen = fullscreen
        self.clicks_to_close = 5

        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans',get_opensans_font_path())
        self.glfont.set_size(32)
        self.glfont.set_color_float((0.2,0.5,0.9,1.0))
        self.glfont.set_align_string(v_align='center')

        # UI Platform tweaks
        if system() == 'Linux':
            self.window_position_default = (0, 0)
        elif system() == 'Windows':
            self.window_position_default = (8, 90)
        else:
            self.window_position_default = (0, 0)

        self.circle_tracker = CircleTracker()
        self.markers = []

    def init_ui(self):
        super().init_ui()
        self.menu.label = "Screen Marker Calibration"

        def get_monitors_idx_list():
            monitors = [glfwGetMonitorName(m) for m in glfwGetMonitors()]
            return range(len(monitors)),monitors

        if self.monitor_idx not in get_monitors_idx_list()[0]:
            logger.warning("Monitor at index %s no longer availalbe using default"%self.monitor_idx)
            self.monitor_idx = 0

        self.menu.append(ui.Info_Text("Calibrate gaze parameters using a screen based animation."))
        self.menu.append(ui.Selector('monitor_idx',self,selection_getter = get_monitors_idx_list,label='Monitor'))
        self.menu.append(ui.Switch('fullscreen',self,label='Use fullscreen'))
        self.menu.append(ui.Slider('marker_scale',self,step=0.1,min=0.5,max=2.0,label='Marker size'))
        self.menu.append(ui.Slider('sample_duration',self,step=1,min=10,max=100,label='Sample duration'))

    def start(self):
        if not self.g_pool.capture.online:
            logger.error("{} requiers world capture video input.".format(self.mode_pretty))
            return
        super().start()
        audio.say("Starting {}".format(self.mode_pretty))
        logger.info("Starting {}".format(self.mode_pretty))

        if self.g_pool.detection_mapping_mode == '3d':
            if self.mode == 'calibration':
                self.sites = [(.5, .5), (0., 1.), (1., 1.), (1., 0.), (0., 0.)]
            else:
                self.sites = [(.25, .5), (.5, .25), (.75, .5), (.5, .75)]
        else:
            if self.mode == 'calibration':
                self.sites = [(.25, .5), (0, .5), (0., 1.), (.5, 1.), (1., 1.),
                              (1., .5), (1., 0.), (.5, 0.), (0., 0.), (.75, .5)]
            else:
                self.sites = [(.5, .5), (.25, .25), (.25, .75), (.75, .75), (.75, .25)]

        self.active_site = self.sites.pop(0)
        self.active = True
        self.ref_list = []
        self.pupil_list = []
        self.clicks_to_close = 5
        self.open_window(self.mode_pretty)

    def open_window(self, title='new_window'):
        if not self._window:
            if self.fullscreen:
                try:
                    monitor = glfwGetMonitors()[self.monitor_idx]
                except:
                    logger.warning("Monitor at index %s no longer availalbe using default"%self.monitor_idx)
                    self.monitor_idx = 0
                    monitor = glfwGetMonitors()[self.monitor_idx]
                width, height, redBits, blueBits, greenBits, refreshRate = glfwGetVideoMode(monitor)
            else:
                monitor = None
                width,height= 640,360

            self._window = glfwCreateWindow(width, height, title, monitor=monitor, share=glfwGetCurrentContext())
            if not self.fullscreen:
                glfwSetWindowPos(self._window, self.window_position_default[0], self.window_position_default[1])

            glfwSetInputMode(self._window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN)

            # Register callbacks
            glfwSetFramebufferSizeCallback(self._window, on_resize)
            glfwSetKeyCallback(self._window, self.on_window_key)
            glfwSetMouseButtonCallback(self._window, self.on_window_mouse_button)
            on_resize(self._window, *glfwGetFramebufferSize(self._window))

            # gl_state settings
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()
            # refresh speed settings
            glfwSwapInterval(0)

            glfwMakeContextCurrent(active_window)

    def on_window_key(self,window, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                self.clicks_to_close = 0

    def on_window_mouse_button(self,window,button, action, mods):
        if action ==GLFW_PRESS:
            self.clicks_to_close -=1

    def stop(self):
        # TODO: redundancy between all gaze mappers -> might be moved to parent class
        audio.say("Stopping {}".format(self.mode_pretty))
        logger.info("Stopping {}".format(self.mode_pretty))
        self.smooth_pos = 0, 0
        self.counter = 0
        self.close_window()
        self.active = False
        self.button.status_text = ''
        if self.mode == 'calibration':
            finish_calibration(self.g_pool, self.pupil_list, self.ref_list)
        elif self.mode == 'accuracy_test':
            self.finish_accuracy_test(self.pupil_list, self.ref_list)
        super().stop()

    def close_window(self):
        if self._window:
            # enable mouse display
            active_window = glfwGetCurrentContext()
            glfwSetInputMode(self._window, GLFW_CURSOR, GLFW_CURSOR_NORMAL)
            glfwDestroyWindow(self._window)
            self._window = None
            glfwMakeContextCurrent(active_window)

    def recent_events(self, events):
        frame = events.get('frame')
        if self.active and frame:
            gray_img = frame.gray

            if self.clicks_to_close <=0:
                self.stop()
                return

            # Update the marker
            self.markers = self.circle_tracker.update(gray_img)
            # Screen marker takes only Ref marker
            self.markers = [marker for marker in self.markers if marker['marker_type'] == 'Ref']

            if len(self.markers):
                # Set the pos to be the center of the first detected marker
                marker_pos = self.markers[0]['img_pos']
                self.pos = self.markers[0]['norm_pos']
            else:
                self.pos = None  # indicate that no reference is detected

            # Check if there are more than one markers
            if len(self.markers) > 1:
                audio.tink()
                logger.warning("{} markers detected. Please remove all the other markers".format(len(self.markers)))

            # only save a valid ref position if within sample window of calibration routine
            on_position = self.lead_in < self.screen_marker_state < (self.lead_in+self.sample_duration)

            if on_position and len(self.markers):
                ref = {}
                ref["norm_pos"] = self.pos
                ref["screen_pos"] = marker_pos
                ref["timestamp"] = frame.timestamp
                self.ref_list.append(ref)

            # Always save pupil positions
            self.pupil_list.extend(events['pupil_positions'])

            if on_position and len(self.markers) and events.get('fixations', []):
                fixation_boost = 5
                self.screen_marker_state = min(
                    self.sample_duration+self.lead_in,
                    self.screen_marker_state+fixation_boost)

            # Animate the screen marker
            if self.screen_marker_state < self.sample_duration+self.lead_in+self.lead_out:
                if len(self.markers) or not on_position:
                    self.screen_marker_state += 1
            else:
                self.screen_marker_state = 0
                if not self.sites:
                    self.stop()
                    return
                self.active_site = self.sites.pop(0)
                logger.debug("Moving screen marker to site at {} {}".format(*self.active_site))

            # use np.arrays for per element wise math
            self.display_pos = np.array(self.active_site)
            self.on_position = on_position
            self.button.status_text = '{}'.format(self.active_site)

        if self._window:
            self.gl_display_in_window()

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """

        # debug mode within world will show green ellipses around detected ellipses
        if self.active:
            for marker in self.markers:
                e = marker['ellipses'][-1]# outermost ellipse
                pts = cv2.ellipse2Poly((int(e[0][0]), int(e[0][1])),
                                       (int(e[1][0]/2), int(e[1][1]/2)),
                                       int(e[-1]), 0, 360, 15)
                draw_polyline(pts, 1, RGBA(0.,1.,0.,1.))
                if len(self.markers) > 1:
                    draw_polyline(pts, 1, RGBA(1., 0., 0., .5), line_type=gl.GL_POLYGON)

    def gl_display_in_window(self):
        active_window = glfwGetCurrentContext()
        if glfwWindowShouldClose(self._window):
            self.close_window()
            return

        glfwMakeContextCurrent(self._window)

        clear_gl_screen()

        hdpi_factor = getHDPIFactor(self._window)
        r = self.marker_scale * hdpi_factor
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        p_window_size = glfwGetFramebufferSize(self._window)
        gl.glOrtho(0, p_window_size[0], p_window_size[1], 0, -1, 1)
        # Switch back to Model View Matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        def map_value(value,in_range=(0,1),out_range=(0,1)):
            ratio = (out_range[1]-out_range[0])/(in_range[1]-in_range[0])
            return (value-in_range[0])*ratio+out_range[0]

        pad = 90 * r
        screen_pos = map_value(self.display_pos[0],out_range=(pad,p_window_size[0]-pad)),map_value(self.display_pos[1],out_range=(p_window_size[1]-pad,pad))
        alpha = interp_fn(self.screen_marker_state,0.,1.,float(self.sample_duration+self.lead_in+self.lead_out),float(self.lead_in),float(self.sample_duration+self.lead_in))

        r2 = 2 * r
        draw_points([screen_pos], size=60*r2, color=RGBA(0., 0., 0., alpha), sharpness=0.9)
        draw_points([screen_pos], size=38*r2, color=RGBA(1., 1., 1., alpha), sharpness=0.8)
        draw_points([screen_pos], size=19*r2, color=RGBA(0., 0., 0., alpha), sharpness=0.55)

        # some feedback on the detection state
        color = RGBA(0., .8, 0., alpha) if len(self.markers) and self.on_position else RGBA(0.8, 0., 0., alpha)
        draw_points([screen_pos], size=3*r2, color=color, sharpness=0.5)

        if self.clicks_to_close <5:
            self.glfont.set_size(int(p_window_size[0]/30.))
            self.glfont.draw_text(p_window_size[0]/2.,p_window_size[1]/4.,'Touch {} more times to cancel {}.'.format(self.clicks_to_close, self.mode_pretty))

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)

    def get_init_dict(self):
        d = {}
        d['fullscreen'] = self.fullscreen
        d['marker_scale'] = self.marker_scale
        d['monitor_idx'] = self.monitor_idx
        return d

    def deinit_ui(self):
        """gets called when the plugin get terminated.
           either voluntarily or forced.
        """
        if self.active:
            self.stop()
        if self._window:
            self.close_window()
        super().deinit_ui()
