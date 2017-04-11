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

from queue import Full, Empty
from multiprocessing import Event, Queue, Process

import logging
logger = logging.getLogger(__name__)


class Global_Container(object):
    pass


def detect_marker_positions(alive_flag, detection_queue, source_path, timestamps_path):
    alive_flag.set()  # mark detection process as alive
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
        while alive_flag.is_set():
            cur_ts = frame.timestamp
            print('\rDetection progress: {:3.0f} %'.format(100 * (cur_ts - min_ts) / (max_ts - min_ts)), end="")

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
                        # if events.get('fixations', []):
                        #     self.counter -= self.fixation_boost
                        if counter <= 0:
                            print("Sampled {} datapoints. Stopping to sample. Looking for steady marker again.".format(counter_max))
                        try:
                            detection_queue.put(ref, timeout=.05)
                        except Full:
                            pass
            # end of tracking logic

            frame = src.get_frame()

    except EndofVideoFileError:
        pass
    alive_flag.clear()


class Offline_Calibration(Plugin):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.ref_positions = []
        self.original_gaze_pos_by_frame = self.g_pool.gaze_positions_by_frame

        source_path = g_pool.capture.source_path
        timestamps_path = os.path.join(g_pool.rec_dir, "world_timestamps.npy")

        self.detect_process_is_alive = Event()
        self.detection_queue = Queue()
        self.detection_process = Process(target=detect_marker_positions,
                                         name='Calibration Marker Detection',
                                         args=(self.detect_process_is_alive,
                                               self.detection_queue,
                                               source_path, timestamps_path))
        self.detection_process.start()
        # self.detect_process_is_alive.wait()  # optional

    def init_gui(self):
        def close():
            self.alive = False
        self.menu = ui.Scrolling_Menu("Offline Calibration", size=(200, 300))
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Button('Close', close))

        # self.menu.append(ui.Button('Redetect', self.redetect))

    def deinit_gui(self):
        if hasattr(self, 'menu'):
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        return {}

    def recent_events(self, events):
        try:
            while True:
                ref_pos = self.detection_queue.get_nowait()
                self.ref_positions.append(ref_pos)
        except Empty:
            pass

    def cleanup(self):
        self.detect_process_is_alive.clear()
        self.detection_process.join()
        self.g_pool.gaze_positions_by_frame = self.original_gaze_pos_by_frame
        self.notify_all({'subject': 'gaze_positions_changed'})
        self.deinit_gui()

