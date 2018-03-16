'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from pyglui.cygl.utils import draw_points_norm, draw_polyline_norm, RGBA
from pyglui.ui import Info_Text
from OpenGL.GL import GL_POLYGON
from glfw import GLFW_PRESS, GLFW_KEY_SPACE, GLFW_KEY_F, glfwSetKeyCallback
import audio
from fingertip_detector import FingertipTracker
from . finish_calibration import finish_calibration
from . calibration_plugin_base import Calibration_Plugin
from plugin import Experimental_Plugin_Base

# logging
import logging
logger = logging.getLogger(__name__)


class Fingertip_Calibration(Experimental_Plugin_Base, Calibration_Plugin):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.pos = None
        self.fingertip_tracker = FingertipTracker()
        self.fingertip = None
        self.press_key = 0

        self.menu = None

    def init_ui(self):
        super().init_ui()
        self.menu.label = 'Fingertip Calibration'
        self.menu.append(Info_Text('Calibrate gaze parameters using your finger tip.'))
        self.menu.append(Info_Text('This is an experimental calibration routine. '
                                   'Move your index finger into your field of view while looking at the fingertip.'
                                   'This plugin is for prototyping and experimentation only. '
                                   'The detection robustness is not production grade. '
                                   'We will put a lot more effort into this going forward but wanted to release the idea and hope for feedback!'))
        self.menu.append(Info_Text('This is a two step process: '
                                   '(1) calibrating for skin tone of the participant '
                                   '(2) collecting calibration samples.'))

    def start(self):
        super().start()
        audio.say('Starting {}'.format(self.mode_pretty))
        logger.info('Starting {}'.format(self.mode_pretty))
        self.active = True
        self.ref_list = []
        self.pupil_list = []
        glfwSetKeyCallback(self.g_pool.main_window, self.on_window_key)
        self.button.status_text = ''
        if self.mode == 'calibration':
            self.press_key = -1
        elif self.mode == 'accuracy_test':
            self.fingertip_tracker.train_done = 3

    def stop(self):
        self.button.status_text = ''
        audio.say('Stopping  {}'.format(self.mode_pretty))
        logger.info('Stopping  {}'.format(self.mode_pretty))
        self.active = False
        if self.mode == 'calibration':
            finish_calibration(self.g_pool, self.pupil_list, self.ref_list)
        elif self.mode == 'accuracy_test':
            self.finish_accuracy_test(self.pupil_list, self.ref_list)
        super().stop()

    def recent_events(self, events):
        """
        gets called once every frame.
        reference positon need to be published to shared_pos
        if no reference was found, publish 0,0
        """
        frame = events.get('frame')

        if self.active and frame:
            recent_pupil_positions = events['pupil_positions']
            # Update the marker
            if self.mode == 'calibration':
                if self.press_key in (-1, 2):
                    self.button.status_text = 'Cover the rectangles with your index finger and then press space'
                elif self.press_key == 1 and self.fingertip_tracker.train_done == 1:
                    self.button.status_text = 'Press space to start calibration'
                elif self.press_key == 1 and self.fingertip_tracker.train_done == 2:
                    self.button.status_text = 'Press C to stop calibration; Press F to re-detect the skin tone if needed'

            self.fingertip = self.fingertip_tracker.update(frame.bgr, self.press_key)
            self.press_key = 0

            if self.fingertip is not None:
                # Set the pos to be the center of the first detected marker
                marker_pos = self.fingertip['screen_pos']
                self.pos = self.fingertip['norm_pos']
                ref = {}
                ref['norm_pos'] = self.pos
                ref['screen_pos'] = marker_pos
                ref['timestamp'] = frame.timestamp
                self.ref_list.append(ref)
            else:
                self.pos = None  # indicate that no reference is detected

            # Always save pupil positions
            for p_pt in recent_pupil_positions:
                self.pupil_list.append(p_pt)
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
            if self.pos is not None:
                draw_points_norm([self.pos], size=15, color=RGBA(0.1,0.8,0.9,1.0))
                p = self.fingertip['norm_rect_points']
                draw_polyline_norm([p[0], p[1], p[2], p[3], p[0]], color=RGBA(0.1,0.8,0.9,0.3), thickness=3)

            if self.mode == 'calibration' and self.fingertip_tracker.train_done in (0, 1):
                for p in self.fingertip_tracker.ROIpts:
                    points = [(x, 1-y) for x in [p[1], p[1]+p[3]] for y in [p[0], p[0]+p[2]]]
                    draw_polyline_norm([points[0], points[1], points[3], points[2]], color=RGBA(0.1,0.9,0.7,1.0), line_type=GL_POLYGON)

    def on_window_key(self, window, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if key == GLFW_KEY_SPACE:
                self.press_key = 1
            elif key == GLFW_KEY_F:
                self.press_key = 2

    def deinit_ui(self):
        """gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self.active:
            self.stop()
        super().deinit_ui()
