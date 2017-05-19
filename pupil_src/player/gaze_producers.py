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
from pyglui import ui
from plugin import Producer_Plugin_Base
from player_methods import correlate_data
from methods import normalize
from video_capture import File_Source, EndofVideoFileError
from circle_detector import find_concetric_circles

from calibration_routines import gaze_mapping_plugins
from calibration_routines.finish_calibration import select_calibration_method
from file_methods import Persistent_Dict

import background_helper as bh
from itertools import chain

import logging
logger = logging.getLogger(__name__)

gaze_mapping_plugins_by_name = {p.__name__: p for p in gaze_mapping_plugins}


class Gaze_Producer_Base(Producer_Plugin_Base):
    uniqueness = 'by_base_class'
    order = .02


class Gaze_From_Recording(Gaze_Producer_Base):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        g_pool.gaze_positions = g_pool.pupil_data['gaze_positions']
        g_pool.gaze_positions_by_frame = correlate_data(g_pool.gaze_positions, g_pool.timestamps)
        self.notify_all({'subject': 'gaze_positions_changed'})
        logger.debug('gaze positions changed')


class Global_Container(object):
    pass


def detect_marker_positions(cmd_pipe, data_pipe, source_path, timestamps_path):
    timestamps = np.load(timestamps_path)
    min_ts = timestamps[0]
    max_ts = timestamps[-1]

    src = File_Source(Global_Container(), source_path, timestamps, timed_playback=False)
    frame = src.get_frame()

    logger.info('Starting calibration marker detection...')

    try:
        while True:
            for event in bh.recent_events(cmd_pipe):
                if event == bh.TERM_SIGNAL:
                    raise RuntimeError()

            progress = 100 * (frame.timestamp - min_ts) / (max_ts - min_ts)
            cmd_pipe.send(('progress', progress))

            gray_img = frame.gray
            markers = find_concetric_circles(gray_img, min_ring_count=3)
            if len(markers) > 0:
                detected = True
                marker_pos = markers[0][0][0]  # first marker innermost ellipse, pos
                pos = normalize(marker_pos, (frame.width, frame.height), flip_y=True)

            else:
                detected = False
                pos = None

            if detected:
                second_ellipse = markers[0][1]
                col_slice = int(second_ellipse[0][0]-second_ellipse[1][0]/2),int(second_ellipse[0][0]+second_ellipse[1][0]/2)
                row_slice = int(second_ellipse[0][1]-second_ellipse[1][1]/2),int(second_ellipse[0][1]+second_ellipse[1][1]/2)
                marker_gray = gray_img[slice(*row_slice),slice(*col_slice)]
                avg = cv2.mean(marker_gray)[0]
                center = marker_gray[int(second_ellipse[1][1])//2, int(second_ellipse[1][0])//2]
                rel_shade = center-avg

                ref = {}
                ref["norm_pos"] = pos
                ref["screen_pos"] = marker_pos
                ref["timestamp"] = frame.timestamp
                ref['index'] = frame.index
                if rel_shade > 30:
                    ref['type'] = 'stop_marker'
                else:
                    ref['type'] = 'calibration_marker'

                data_pipe.send(ref)
            frame = src.get_frame()

    except (EndofVideoFileError, RuntimeError, EOFError, OSError, BrokenPipeError):
        pass
    finally:
        cmd_pipe.send(('finished',))  # one-element tuple required
        cmd_pipe.close()
        data_pipe.close()


def map_pupil_positions(cmd_pipe, data_pipe, pupil_list, gaze_mapper_cls_name, kwargs):
    try:
        gaze_mapper_cls = gaze_mapping_plugins_by_name[gaze_mapper_cls_name]
        gaze_mapper = gaze_mapper_cls(Global_Container(), **kwargs)
        for idx, datum in enumerate(pupil_list):
            for event in bh.recent_events(cmd_pipe):
                if event == bh.TERM_SIGNAL:
                    raise RuntimeError()

            mapped_gaze = gaze_mapper.on_pupil_datum(datum)
            if mapped_gaze:
                data_pipe.send(mapped_gaze)
                progress = 100 * (idx+1)/len(pupil_list)
                cmd_pipe.send(('progress', progress))

    except (RuntimeError, EOFError, OSError, BrokenPipeError):
        pass
    finally:
        cmd_pipe.send(('finished',))  # one-element tuple required
        cmd_pipe.close()
        data_pipe.close()


class Offline_Calibration(Gaze_Producer_Base):
    def __init__(self, g_pool, mapping_cls_name='Dummy_Gaze_Mapper', mapping_args={}, detection_mapping_mode='3d'):
        super().__init__(g_pool)
        self.mapping_cls_name = mapping_cls_name
        self.mapping_args = mapping_args

        self.gaze_positions = []
        self.mapping_progress = 0.
        self.mapping_proxy = None

        self.load_previously_detected_markers()
        self.detection_proxy = None
        if self.ref_positions:
            self.detection_progress = 100.0
        else:
            self.detection_progress = 0.0
            self.start_detection_task()

        self.g_pool.detection_mapping_mode = detection_mapping_mode
        self.g_pool.active_calibration_plugin = self
        self.start_mapping_task()

    def start_detection_task(self, *_):
        # cancel current detection if running
        bh.cancel_background_task(self.detection_proxy, False)

        self.ref_positions = []
        source_path = self.g_pool.capture.source_path
        timestamps_path = os.path.join(self.g_pool.rec_dir, "world_timestamps.npy")

        self.detection_proxy = bh.start_background_task(detect_marker_positions,
                                                        name='Calibration Marker Detection',
                                                        args=(source_path, timestamps_path))

    def start_mapping_task(self, *_):
        # cancel current mapping if running
        self.mapping_progress = 0.
        bh.cancel_background_task(self.mapping_proxy, False)

        self.gaze_positions = []
        pupil_list = self.g_pool.pupil_data['pupil_positions']

        self.mapping_proxy = bh.start_background_task(map_pupil_positions,
                                                      name='Gaze Mapping',
                                                      args=(pupil_list, self.mapping_cls_name, self.mapping_args))

    def load_previously_detected_markers(self):
        loaded = Persistent_Dict(os.path.join(self.g_pool.rec_dir, 'circle_markers'))
        self.ref_positions = loaded.get('offline_detected', [])

    def save_detected_markers(self):
        storage = Persistent_Dict(os.path.join(self.g_pool.rec_dir, 'circle_markers'))
        storage['offline_detected'] = self.ref_positions
        storage.save()

    def init_gui(self):
        self.menu = ui.Scrolling_Menu("Offline Calibration", pos=(-660, 20), size=(300, 500))
        self.g_pool.gui.append(self.menu)

        self.menu.append(ui.Info_Text('"Detection" searches for calibration markers in the world video.'))
        self.menu.append(ui.Button('Redetect', self.start_detection_task))
        slider = ui.Slider('detection_progress', self, label='Detection Progress')
        slider.display_format = '%3.0f%%'
        slider.read_only = True
        self.menu.append(slider)

        self.menu.append(ui.Info_Text('"Mapping" recalculates all gaze positions using the current gaze mapper.'))
        self.menu.append(ui.Selector('detection_mapping_mode', self.g_pool, selection=['2d', '3d'],
                                     label='Mapping Mode'))
        self.menu.append(ui.Button('Remap', self.start_mapping_task))
        slider = ui.Slider('mapping_progress', self, label='Mapping Progress')
        slider.display_format = '%3.0f%%'
        slider.read_only = True
        self.menu.append(slider)

    def deinit_gui(self):
        if hasattr(self, 'menu'):
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        return {'mapping_cls_name': self.mapping_cls_name,
                'mapping_args': self.mapping_args,
                'detection_mapping_mode': self.g_pool.detection_mapping_mode}

    def on_notify(self, notification):
        subject = notification['subject']
        if subject == 'pupil_positions_changed' and not self.detection_proxy:
            self.calibrate()  # do not calibrate while detection task is still running

    def recent_events(self, events):
        if self.detection_proxy:
            for ref_pos in bh.recent_events(self.detection_proxy.data):
                self.ref_positions.append(ref_pos)
            for msg in bh.recent_events(self.detection_proxy.cmd):
                if msg[0] == 'progress':
                    self.detection_progress = msg[1]
                elif msg[0] == 'finished':
                    self.detection_proxy = None
                    self.calibrate()

        if self.mapping_proxy:
            for mapped_gaze in bh.recent_events(self.mapping_proxy.data):
                self.gaze_positions.extend(mapped_gaze)
            for msg in bh.recent_events(self.mapping_proxy.cmd):
                if msg[0] == 'progress':
                    self.mapping_progress = msg[1]
                elif msg[0] == 'finished':
                    self.mapping_proxy = None
                    self.finish_mapping()

    def calibrate(self):
        if not self.ref_positions:
            logger.error('No markers have been found. Cannot calibrate.')
            return

        first_idx = self.ref_positions[0]['index']
        last_idx = self.ref_positions[-1]['index']
        pupil_list = list(chain(*self.g_pool.pupil_positions_by_frame[first_idx:last_idx]))
        logger.info('Calibrating...')
        method, result = select_calibration_method(self.g_pool, pupil_list, self.ref_positions)
        if result['subject'] != 'calibration.failed':
            logger.info('Offline calibration successful. Starting mapping using {}.'.format(method))
            self.mapping_cls_name = result['name']
            self.mapping_args = result['args']
            self.start_mapping_task()
        self.save_detected_markers()

    def finish_mapping(self):
        self.g_pool.gaze_positions = self.gaze_positions
        self.g_pool.gaze_positions_by_frame = correlate_data(self.gaze_positions, self.g_pool.timestamps)
        self.notify_all({'subject': 'gaze_positions_changed'})

    def cleanup(self):
        bh.cancel_background_task(self.detection_proxy)
        bh.cancel_background_task(self.mapping_proxy)
        self.deinit_gui()
        self.g_pool.active_calibration_plugin = None


gaze_producers = [Gaze_From_Recording, Offline_Calibration]
