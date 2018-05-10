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
import zmq
import zmq_tools
import numpy as np
from itertools import chain
from plugin import Producer_Plugin_Base
from pyglui import ui
from player_methods import correlate_data
from file_methods import load_object, save_object

from pyglui.pyfontstash import fontstash as fs
from pyglui.cygl.utils import *
import OpenGL.GL as gl
import gl_utils

import pupil_detectors  # trigger module compilation

import logging
logger = logging.getLogger(__name__)

right_color = RGBA(0.9844, 0.5938, 0.4023, 1.)
left_color = RGBA(0.668, 0.6133, 0.9453, 1.)


class Empty(object):
        pass


class Pupil_Producer_Base(Producer_Plugin_Base):
    uniqueness = 'by_base_class'
    order = 0.01
    icon_chr = chr(0xec12)
    icon_font = 'pupil_icons'

    def init_ui(self):
        self.add_menu()

        pupil_producer_plugins = [p for p in self.g_pool.plugin_by_name.values() if issubclass(p, Pupil_Producer_Base)]
        pupil_producer_plugins.sort(key=lambda p: p.__name__)

        self.menu_icon.order = 0.29

        def open_plugin(p):
            self.notify_all({'subject': 'start_plugin', 'name': p.__name__})

        # We add the capture selection menu
        self.menu.append(ui.Selector(
                                'pupil_producer',
                                setter=open_plugin,
                                getter=lambda: self.__class__,
                                selection=pupil_producer_plugins,
                                labels=[p.__name__.replace('_', ' ') for p in pupil_producer_plugins],
                                label='Pupil Producers'
                            ))

        self.cache = {}
        self.cache_pupil_timeline_data('diameter')
        self.cache_pupil_timeline_data('confidence')

        self.glfont = fs.Context()
        self.glfont.add_font('opensans', ui.get_opensans_font_path())
        self.glfont.set_font('opensans')

        self.dia_timeline = ui.Timeline('Pupil Diameter [px]', self.draw_pupil_diameter, self.draw_dia_legend)
        self.conf_timeline = ui.Timeline('Pupil Confidence', self.draw_pupil_conf, self.draw_conf_legend)
        self.g_pool.user_timelines.append(self.dia_timeline)
        self.g_pool.user_timelines.append(self.conf_timeline)

    def on_notify(self, notification):
        if notification['subject'] == 'pupil_positions_changed':
            self.cache_pupil_timeline_data('diameter')
            self.cache_pupil_timeline_data('confidence')
            self.dia_timeline.refresh()
            self.conf_timeline.refresh()

    def deinit_ui(self):
        self.remove_menu()
        self.g_pool.user_timelines.remove(self.dia_timeline)
        self.g_pool.user_timelines.remove(self.conf_timeline)
        self.dia_timeline = None
        self.conf_timeline = None

    def recent_events(self, events):
        if 'frame' in events:
            frm_idx = events['frame'].index
            events['pupil_positions'] = self.g_pool.pupil_positions_by_frame[frm_idx]

    def cache_pupil_timeline_data(self, key):
        t0, t1 = self.g_pool.timestamps[0], self.g_pool.timestamps[-1]
        if not self.g_pool.pupil_positions:
            self.cache[key] = {'left': [], 'right': [],
                               'xlim': [t0, t1], 'ylim': [0, 1]}
        else:
            right = [(pp['timestamp'], pp[key]) for pp in self.g_pool.pupil_positions if pp['id'] == 0]
            left = [(pp['timestamp'], pp[key]) for pp in self.g_pool.pupil_positions if pp['id'] == 1]

            # max_val must not be 0, else gl will crash
            max_val = max(chain((pp[1] for pp in right), (pp[1] for pp in left))) or 1

            self.cache[key] = {'left': left, 'right': right,
                               'xlim': [t0, t1], 'ylim': [0, max_val]}

    def draw_pupil_diameter(self, width, height, scale):
        self.draw_pupil_data('diameter', width, height, scale)

    def draw_pupil_conf(self, width, height, scale):
        self.draw_pupil_data('confidence', width, height, scale)

    def draw_pupil_data(self, key, width, height, scale):
        right = self.cache[key]['right']
        left = self.cache[key]['left']

        with gl_utils.Coord_System(*self.cache[key]['xlim'], *self.cache[key]['ylim']):
            draw_points(right, size=2.*scale, color=right_color)
            draw_points(left, size=2.*scale, color=left_color)

    def draw_dia_legend(self, width, height, scale):
        self.draw_legend(self.dia_timeline.label, width, height, scale)

    def draw_conf_legend(self, width, height, scale):
        self.draw_legend(self.conf_timeline.label, width, height, scale)

    def draw_legend(self, label, width, height, scale):
        self.glfont.push_state()
        self.glfont.set_align_string(v_align='right', h_align='top')
        self.glfont.set_size(15. * scale)
        self.glfont.draw_text(width, 0, label)

        legend_height = 13. * scale
        pad = 10 * scale
        self.glfont.draw_text(width / 2, legend_height, 'left')
        self.glfont.draw_text(width, legend_height, 'right')

        self.glfont.pop_state()

        draw_polyline([(pad, 1.5 * legend_height),
                       (width / 4, 1.5 * legend_height)],
                      color=left_color, line_type=gl.GL_LINES, thickness=4.*scale)
        draw_polyline([(width / 2 + pad, 1.5 * legend_height),
                       (width * 3 / 4, 1.5 * legend_height)],
                      color=right_color, line_type=gl.GL_LINES, thickness=4.*scale)


class Pupil_From_Recording(Pupil_Producer_Base):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        g_pool.pupil_positions = g_pool.pupil_data['pupil_positions']
        g_pool.pupil_positions_by_frame = correlate_data(g_pool.pupil_positions, g_pool.timestamps)
        self.notify_all({'subject': 'pupil_positions_changed'})
        logger.debug('pupil positions changed')

    def init_ui(self):
        super().init_ui()
        self.menu.label = "Pupil Data From Recording"
        self.menu.append(ui.Info_Text('Currently, pupil positions are loaded from the recording.'))


class Offline_Pupil_Detection(Pupil_Producer_Base):
    """docstring for Offline_Pupil_Detection"""
    session_data_version = 1

    def __init__(self, g_pool):
        super().__init__(g_pool)
        zmq_ctx = zmq.Context()
        self.data_sub = zmq_tools.Msg_Receiver(zmq_ctx, g_pool.ipc_sub_url, topics=('pupil','notify.file_source.video_finished'))

        self.data_dir = os.path.join(g_pool.rec_dir, 'offline_data')
        os.makedirs(self.data_dir, exist_ok=True)
        try:
            session_data = load_object(os.path.join(self.data_dir, 'offline_pupil_data'))
            assert session_data.get('version') == self.session_data_version
        except (AssertionError, FileNotFoundError):
            session_data = {}
            session_data["detection_method"] = '3d'
            session_data['pupil_positions'] = []
            session_data['detection_status'] = ["unknown", "unknown"]
        self.detection_method = session_data["detection_method"]
        self.pupil_positions = {pp['timestamp']: pp for pp in session_data['pupil_positions']}
        self.detection_status = session_data['detection_status']
        self.eye_video_loc = [None, None]
        self.eye_frame_num = [0, 0]
        for pp in self.pupil_positions.values():
            self.eye_frame_num[pp['id']] += 1

        self.pause_switch = None
        self.detection_paused = False

        # start processes
        for eye_id in range(2):
            if self.detection_status[eye_id] != 'complete':
                self.start_eye_process(eye_id)

        # either we did not start them or they failed to start (mono setup etc)
        # either way we are done and can publish
        if self.eye_video_loc == [None, None]:
            self.correlate_publish()

    def start_eye_process(self, eye_id):
        potential_locs = [os.path.join(self.g_pool.rec_dir, 'eye{}{}'.format(eye_id, ext)) for ext in ('.mjpeg', '.mp4', '.mkv')]
        existing_locs = [loc for loc in potential_locs if os.path.exists(loc)]
        timestamps_path = os.path.join(self.g_pool.rec_dir, 'eye{}_timestamps.npy'.format(eye_id))

        if not existing_locs:
            logger.error("no eye video for eye '{}' found.".format(eye_id))
            self.detection_status[eye_id] = "No eye video found."
            return
        if not os.path.exists(timestamps_path):
            logger.error("no timestamps for eye video for eye '{}' found.".format(eye_id))
            self.detection_status[eye_id] = "No eye video found."
            return

        video_loc = existing_locs[0]
        self.eye_frame_num[eye_id] = len(np.load(timestamps_path))

        capure_settings = 'File_Source', {
            'source_path': video_loc,
            'timing': None
        }
        self.notify_all({'subject': 'eye_process.should_start', 'eye_id': eye_id,
                         'overwrite_cap_settings': capure_settings})
        self.eye_video_loc[eye_id] = video_loc
        self.detection_status[eye_id] = "Detecting..."

    def stop_eye_process(self, eye_id):
        self.notify_all({'subject': 'eye_process.should_stop', 'eye_id': eye_id})
        self.eye_video_loc[eye_id] = None

    def recent_events(self, events):
        super().recent_events(events)
        while self.data_sub.new_data:
            topic, payload = self.data_sub.recv()
            if topic.startswith('pupil.'):
                self.pupil_positions[payload['timestamp']] = payload
            elif payload['subject'] == 'file_source.video_finished':
                if self.eye_video_loc[0] == payload['source_path']:
                    logger.debug("eye 0 process complete")
                    self.detection_status[0] = "complete"
                    self.stop_eye_process(0)
                elif self.eye_video_loc[1] == payload['source_path']:
                    logger.debug("eye 1 process complete")
                    self.detection_status[1] = "complete"
                    self.stop_eye_process(1)
                if self.eye_video_loc == [None, None]:
                    self.correlate_publish()
        total = sum(self.eye_frame_num)
        self.menu_icon.indicator_stop = len(self.pupil_positions) / total if total else 0.

    def correlate_publish(self):
        self.g_pool.pupil_positions = sorted(self.pupil_positions.values(), key=lambda pp: pp['timestamp'])
        self.g_pool.pupil_positions_by_frame = correlate_data(list(self.pupil_positions.values()), self.g_pool.timestamps)
        self.notify_all({'subject': 'pupil_positions_changed'})
        logger.debug('pupil positions changed')
        self.save_offline_data()

    def on_notify(self, notification):
        super().on_notify(notification)
        if notification['subject'] == 'eye_process.started':
            self.set_detection_mapping_mode(self.detection_method)
        elif notification['subject'] == 'eye_process.stopped':
            self.eye_video_loc[notification['eye_id']] = None

    def cleanup(self):
        self.stop_eye_process(0)
        self.stop_eye_process(1)
        # close sockets before context is terminated
        self.data_sub = None
        self.save_offline_data()

    def save_offline_data(self):
        session_data = {}
        session_data["detection_method"] = self.detection_method
        session_data['pupil_positions'] = list(self.pupil_positions.values())
        session_data['detection_status'] = self.detection_status
        session_data['version'] = self.session_data_version
        cache_path = os.path.join(self.data_dir, 'offline_pupil_data')
        save_object(session_data, cache_path)
        logger.info('Cached detected pupil data to {}'.format(cache_path))

    def redetect(self):
        self.pupil_positions.clear()  # delete previously detected pupil positions
        self.g_pool.pupil_positions_by_frame = [[] for x in self.g_pool.timestamps]
        self.detection_finished_flag = False
        self.detection_paused = False
        for eye_id in range(2):
            if self.eye_video_loc[eye_id] is None:
                self.start_eye_process(eye_id)
            else:
                self.notify_all({'subject': 'file_source.seek', 'frame_index': 0,
                                 'source_path': self.eye_video_loc[eye_id]})

    def set_detection_mapping_mode(self, new_mode):
        n = {'subject': 'set_detection_mapping_mode', 'mode': new_mode}
        self.notify_all(n)
        self.redetect()
        self.detection_method = new_mode

    def init_ui(self):
        super().init_ui()
        self.menu.label = "Offline Pupil Detector"
        self.menu.append(ui.Info_Text('Detects pupil positions from the recording\'s eye videos.'))
        self.menu.append(ui.Selector('detection_method', self, label='Detection Method',
                                     selection=['2d', '3d'], setter=self.set_detection_mapping_mode))
        self.menu.append(ui.Switch('detection_paused', self, label='Pause detection'))
        self.menu.append(ui.Button('Redetect', self.redetect))
        self.menu.append(ui.Text_Input("0", label='eye0:', getter=lambda: self.detection_status[0], setter=lambda _: _))
        self.menu.append(ui.Text_Input("1", label='eye1:', getter=lambda: self.detection_status[1], setter=lambda _: _))

        def detection_progress():
            total = sum(self.eye_frame_num)
            return 100 * len(self.pupil_positions) / total if total else 0.

        progress_slider = ui.Slider('detection_progress',
                                    label='Detection Progress',
                                    getter=detection_progress,
                                    setter=lambda _: _)
        progress_slider.display_format = '%3.0f%%'
        self.menu.append(progress_slider)

    @property
    def detection_paused(self):
        return self._detection_paused

    @detection_paused.setter
    def detection_paused(self, should_pause):
        self._detection_paused = should_pause
        for eye_id in range(2):
            if self.eye_video_loc[eye_id] is not None:
                subject = 'file_source.' + ('should_pause' if should_pause else 'should_play')
                self.notify_all({'subject': subject, 'source_path': self.eye_video_loc[eye_id]})
