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
import numpy as np
from pyglui import ui
from plugin import Plugin
from player_methods import correlate_data
from methods import normalize, denormalize
from video_capture import File_Source, EndofVideoFileError
from circle_detector import find_concetric_circles

from calibration_routines import Dummy_Gaze_Mapper
from calibration_routines.finish_calibration import finish_calibration

import background_helper as bh
from itertools import chain

import logging
logger = logging.getLogger(__name__)


class Global_Container(object):
    pass


def detect_marker_positions(cmd_pipe, data_pipe, source_path, timestamps_path):
    timestamps = np.load(timestamps_path)
    min_ts = timestamps[0]
    max_ts = timestamps[-1]

    src = File_Source(Global_Container(), source_path, timestamps, timed_playback=False)
    frame = src.get_frame()

    smooth_pos = 0., 0.
    smooth_vel = 0.
    sample_site = (-2, -2)
    counter = 0
    counter_max = 3

    logger.info('Starting calibration marker detection...')

    try:
        while True:
            for event in bh.recent_events(cmd_pipe):
                if event == bh.TERM_SIGNAL:
                    raise RuntimeError()

            progress = 100 * (frame.timestamp - min_ts) / (max_ts - min_ts)
            cmd_pipe.send('progress')
            cmd_pipe.send(progress)

            gray_img = frame.gray
            markers = find_concetric_circles(gray_img, min_ring_count=3)
            if len(markers) > 0:
                detected = True
                marker_pos = markers[0][0][0]  # first marker innermost ellipse, pos
                pos = normalize(marker_pos, (frame.width, frame.height), flip_y=True)
            else:
                detected = False
                pos = None

            # tracking logic
            if detected:
                # calculate smoothed manhattan velocity
                smoother = 0.3
                smooth_pos = np.array(smooth_pos)
                pos = np.array(pos)
                new_smooth_pos = smooth_pos + smoother*(pos-smooth_pos)
                smooth_vel_vec = new_smooth_pos - smooth_pos
                smooth_pos = new_smooth_pos
                smooth_pos = list(smooth_pos)
                # manhattan distance for velocity
                new_vel = abs(smooth_vel_vec[0])+abs(smooth_vel_vec[1])
                smooth_vel = smooth_vel + smoother * (new_vel - smooth_vel)

                # distance to last sampled site
                sample_ref_dist = smooth_pos - np.array(sample_site)
                sample_ref_dist = abs(sample_ref_dist[0])+abs(sample_ref_dist[1])

                # start counter if ref is resting in place and not at last sample site
                if counter <= 0:

                    if smooth_vel < 0.01 and sample_ref_dist > 0.1:
                        sample_site = smooth_pos
                        counter = counter_max
                        print("Steady marker found. Starting to sample {} datapoints".format(counter_max))

                if counter > 0:
                    if smooth_vel > 0.01:
                        counter = 0
                        print("Marker moved too quickly: Aborted sample. Sampled {} datapoints. Looking for steady marker again.".format(counter_max-counter))
                    else:
                        counter -= 1
                        ref = {}
                        ref["norm_pos"] = pos
                        ref["screen_pos"] = marker_pos
                        ref["timestamp"] = frame.timestamp
                        ref['index'] = frame.index
                        # if events.get('fixations', []):
                        #     self.counter -= self.fixation_boost
                        if counter <= 0:
                            print("Sampled {} datapoints. Stopping to sample. Looking for steady marker again.".format(counter_max))
                        data_pipe.send(ref)
            # end of tracking logic

            frame = src.get_frame()

    except (EndofVideoFileError, RuntimeError, EOFError, OSError, BrokenPipeError):
        pass
    finally:
        cmd_pipe.send('finished')
        cmd_pipe.close()
        data_pipe.close()


class Offline_Calibration(Plugin):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.ref_positions = []
        self.original_gaze_pos_by_frame = self.g_pool.gaze_positions_by_frame

        self.g_pool.detection_mapping_mode = '3d'
        self.g_pool.plugins.add(Dummy_Gaze_Mapper)
        self.g_pool.active_calibration_plugin = self

        self.detection_proxy = None
        self.detection_progress = 0.0
        self.start_detection_task()

    def start_detection_task(self):
        # cancel current detection if running
        bh.cancel_background_task(self.detection_proxy, False)

        source_path = self.g_pool.capture.source_path
        timestamps_path = os.path.join(self.g_pool.rec_dir, "world_timestamps.npy")

        self.detection_proxy = bh.start_background_task(detect_marker_positions,
                                                        name='Calibration Marker Detection',
                                                        args=(source_path, timestamps_path))

    def init_gui(self):
        if not hasattr(self.g_pool, 'sidebar'):
            # Will be required when loading gaze mappers
            self.g_pool.sidebar = ui.Scrolling_Menu("Sidebar", pos=(-700, 20), size=(300, 500))
            self.g_pool.gui.append(self.g_pool.sidebar)

        def close():
            self.alive = False
        self.menu = ui.Growing_Menu("Offline Calibration")
        self.g_pool.sidebar.insert(0, self.menu)
        self.menu.append(ui.Button('Close', close))

        progress_slider = ui.Slider('detection_progress', self, label='Detection Progress')
        progress_slider.display_format = '%3.0f%%'
        progress_slider.read_only = True
        self.menu.append(progress_slider)
        # self.menu.append(ui.Button('Redetect', self.redetect))

    def deinit_gui(self):
        if hasattr(self, 'menu'):
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        return {}

    def on_notify(self, notification):
        if notification['subject'] == 'pupil_positions_changed' and not self.detection_proxy:
            self.calibrate()  # do not calibrate while detection task is still running
        elif notification['subject'] == 'calibration.successful':
            logger.info('Offline calibration successful. Starting mapping...')

    def recent_events(self, events):
        if self.detection_proxy:
            for ref_pos in bh.recent_events(self.detection_proxy.data):
                self.ref_positions.append(ref_pos)
            for msg in bh.recent_events(self.detection_proxy.cmd):
                if msg == 'progress':
                    self.detection_progress = self.detection_proxy.cmd.recv()
                elif msg == 'finished':
                    self.detection_proxy = None
                    self.calibrate()

    def calibrate(self):
        if not self.ref_positions:
            logger.error('No markers have been found. Cannot calibrate.')
            return

        first_idx = self.ref_positions[0]['index']
        last_idx = self.ref_positions[-1]['index']
        pupil_list = list(chain(*self.g_pool.pupil_positions_by_frame[first_idx:last_idx]))
        finish_calibration(self.g_pool, pupil_list, self.ref_positions)

    def cleanup(self):
        bh.cancel_background_task(self.detection_proxy)
        self.g_pool.gaze_positions_by_frame = self.original_gaze_pos_by_frame
        self.notify_all({'subject': 'gaze_positions_changed'})
        self.deinit_gui()
        self.g_pool.active_gaze_mapping_plugin.alive = False
        del self.g_pool.detection_mapping_mode
        del self.g_pool.active_calibration_plugin
