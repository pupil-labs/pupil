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
from collections import namedtuple

import logging
logger = logging.getLogger(__name__)

gaze_mapping_plugins_by_name = {p.__name__: p for p in gaze_mapping_plugins}


Fake_Capture = namedtuple('Fake_Capture', ['frame_size'])
Fake_Pool = namedtuple('Fake_Pool', ['app', 'get_timestamp', 'capture', 'detection_mapping_mode'])


def setup_fake_pool(frame_size, detection_mode):
    return Fake_Pool('player', lambda: 0., Fake_Capture(frame_size), detection_mode)


def parse_range(range_str, upper_bound):
    range_split = range_str.split('-')
    start = max(0, int(range_split[0]))  # left bound at 0
    end = min(int(range_split[1]), upper_bound)  # right bound at max_ts
    return slice(min(start, end), max(start, end))  # left bound <= right bound


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


def detect_marker_positions(source_path, timestamps_path):
    timestamps = np.load(timestamps_path)
    min_ts = timestamps[0]
    max_ts = timestamps[-1]

    try:
        src = File_Source(Global_Container(), source_path, timestamps, timed_playback=False)
        frame = src.get_frame()

        logger.info('Starting calibration marker detection...')

        while True:

            progress = 100 * (frame.timestamp - min_ts) / (max_ts - min_ts)

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
                marker_gray = gray_img[slice(*row_slice), slice(*col_slice)]
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

                yield progress, ref
            else:
                yield progress, None

            frame = src.get_frame()
    except EndofVideoFileError:
        pass


def calibrate_and_map(g_pool, ref_list, calib_list, map_list):
    method, result = select_calibration_method(g_pool, calib_list, ref_list)
    if result['subject'] != 'calibration.failed':
        logger.info('Offline calibration successful. Starting mapping {} pupil positions using {}.'.format(len(map_list), method))
        name, args = result['name'], result['args']
        gaze_mapper_cls = gaze_mapping_plugins_by_name[name]
        gaze_mapper = gaze_mapper_cls(Global_Container(), **args)

        for idx, datum in enumerate(map_list):
            mapped_gaze = gaze_mapper.on_pupil_datum(datum)
            if mapped_gaze:
                progress = 100 * (idx+1)/len(map_list)
                yield progress, mapped_gaze


class Offline_Calibration(Gaze_Producer_Base):
    def __init__(self, g_pool):
        # mapping_cls_name='Dummy_Gaze_Mapper', mapping_args={},
        super().__init__(g_pool)
        self.menu = None

        result_dir = os.path.join(g_pool.rec_dir, 'offline_results')
        os.makedirs(result_dir, exist_ok=True)
        self.persistent = Persistent_Dict(os.path.join(result_dir, 'offline_calibration'))
        self.persistent['version'] = 2

        if 'sections' in self.persistent:
            self.sections = self.persistent['sections']
            for sec in self.sections:
                sec['progress'] = '{:3.0f}%'.format(0.)
        else:
            init_range = '0-{}'.format(len(g_pool.timestamps))
            self.sections = [self.create_section(init_range, init_range)]
            self.persistent['sections'] = self.sections
        self.bg_tasks = [None] * len(self.sections)
        self.interim_data = [[]] * len(self.sections)

        self.ref_positions = self.persistent.get('ref_positions', [])
        self.ref_positions_by_frame = correlate_data(self.ref_positions, self.g_pool.timestamps)

        self.detection_proxy = None
        if self.ref_positions:
            self.detection_progress = 100.0
            self.calibrate()
        else:
            self.detection_progress = 0.0
            self.start_detection_task()

    def append_section(self):
        init_range = '0-{}'.format(len(self.g_pool.timestamps))
        sec = self.create_section(init_range, init_range)
        self.sections.append(sec)
        self.interim_data.append([])
        self.bg_tasks.append(None)

        if self.menu is not None:
            self.append_section_menu(sec, collapsed=False)

    def create_section(self, calib_range, map_range):
        return {'calibration_range': calib_range,
                'mapping_range': map_range,
                'progress': '{:3.0f}%'.format(0.),
                'label': 'Unnamed'}

    def start_detection_task(self, *_):
        # cancel current detection if running
        if self.detection_proxy is not None:
            self.detection_proxy.cancel()

        self.ref_positions = []
        source_path = self.g_pool.capture.source_path
        timestamps_path = os.path.join(self.g_pool.rec_dir, "world_timestamps.npy")

        self.detection_proxy = bh.Task_Proxy('Calibration Marker Detection',
                                             detect_marker_positions,
                                             args=(source_path, timestamps_path))

    def save_detected_markers(self):
        self.persistent['ref_positions'] = self.ref_positions
        self.persistent.save()

    def init_gui(self):
        self.menu = ui.Scrolling_Menu("Offline Calibration", pos=(-660, 20), size=(300, 500))
        self.g_pool.gui.append(self.menu)

        self.menu.append(ui.Info_Text('"Detection" searches for calibration markers in the world video.'))
        self.menu.append(ui.Button('Redetect', self.start_detection_task))
        slider = ui.Slider('detection_progress', self, label='Detection Progress')
        slider.display_format = '%3.0f%%'
        slider.read_only = True
        self.menu.append(slider)

        self.menu.append(ui.Button('Add section', self.append_section))
        self.menu_section_start = len(self.menu)

        for sec in self.sections:
            self.append_section_menu(sec)

    def append_section_menu(self, sec, collapsed=True):
        section_menu = ui.Growing_Menu('Section "{}"'.format(sec['label']))
        section_menu.collapsed = collapsed

        def set_label(val):
            sec['label'] = val
            section_menu.label = 'Section "{}"'.format(val)

        section_menu.append(ui.Text_Input('label', sec, label='Label', setter=set_label))

        max_ts = len(self.g_pool.timestamps)

        def make_validate_fn(sec, key):
            def validate(range_str):
                try:
                    range_ = parse_range(range_str, max_ts)
                except (ValueError, IndexError):
                    pass  # value error if not parsable by int(), index error of not given 2 ints
                else:
                    sec[key] = '{}-{}'.format(range_.start, range_.stop)
            return validate

        def make_calibrate_fn(sec):
            def calibrate():
                self.calibrate(sec)
            return calibrate

        def make_remove_fn(sec):
            def remove():
                idx = self.sections.index(sec)
                if self.bg_tasks[idx] is not None:
                    self.bg_tasks.cancel()

                del self.bg_tasks[idx]
                del self.interim_data[idx]
                del self.sections[idx]
                del self.menu[self.menu_section_start + idx]
                self.recalculate_gaze_positions()
            return remove

        section_menu.append(ui.Text_Input('calibration_range', sec, label='Calibration range',
                                          setter=make_validate_fn(sec, 'calibration_range')))
        section_menu.append(ui.Text_Input('mapping_range', sec, label='Mapping range',
                                          setter=make_validate_fn(sec, 'mapping_range')))

        section_menu.append(ui.Button('Recalibrate', make_calibrate_fn(sec)))
        section_menu.append(ui.Text_Input('progress', sec, label='Mapping progress', setter=lambda _: _))
        section_menu.append(ui.Button('Remove section', make_remove_fn(sec)))
        self.menu.append(section_menu)

    def deinit_gui(self):
        if hasattr(self, 'menu'):
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        return {}

    def on_notify(self, notification):
        subject = notification['subject']
        if subject == 'pupil_positions_changed' and self.detection_proxy is None:
            self.calibrate()  # do not calibrate while detection task is still running

    def recent_events(self, events):
        if self.detection_proxy:
            recent = [d for d in self.detection_proxy.fetch()]

            if recent:
                progress, data = zip(*recent)
                self.ref_positions.extend([d for d in data if d is not None])
                self.detection_progress = progress[-1]

            if self.detection_proxy.completed:
                self.ref_positions_by_frame = correlate_data(self.ref_positions, self.g_pool.timestamps)
                self.save_detected_markers()
                self.detection_progress = 100.
                self.detection_proxy = None
                self.calibrate()

        for idx, proxy in enumerate(self.bg_tasks):
            if proxy is not None:

                recent = [d for d in proxy.fetch()]
                if recent:
                    progress, data = zip(*recent)
                    # data is a list of list since the mapper returns its results as list with 0-n gaze positions
                    self.interim_data[idx].extend(chain(*data))
                    self.sections[idx]['progress'] = '{:3.0f}%'.format(progress[-1])

                if proxy.completed:
                    self.recalculate_gaze_positions()
                    self.bg_tasks[idx] = None

    def recalculate_gaze_positions(self):
        self.g_pool.gaze_positions = sorted(chain(*self.interim_data), key=lambda d: d['timestamp'])
        self.g_pool.gaze_positions_by_frame = correlate_data(self.g_pool.gaze_positions, self.g_pool.timestamps)
        self.notify_all({'subject': 'gaze_positions_changed'})

    def calibrate(self, section=None):
        if not self.ref_positions:
            logger.error('No markers have been found. Cannot calibrate.')
            return

        # calibrate given section or all known sections
        sections_to_calibrate = (section,) if section is not None else self.sections
        for sec in sections_to_calibrate:
            idx = self.sections.index(sec )
            if self.bg_tasks[idx] is not None:
                self.bg_tasks[idx].cancel()

            sec['progress'] = '{:3.0f}%'.format(0.)
            self.interim_data[idx] = []  # reset interim buffer for given section

            calib_slc = parse_range(sec['calibration_range'], len(self.g_pool.timestamps))
            calib_list = list(chain(*self.g_pool.pupil_positions_by_frame[calib_slc]))
            ref_list = list(chain(*self.ref_positions_by_frame[calib_slc]))

            if not calib_list or not ref_list:
                logger.error('There is not enough data to calibrate section "{}"'.format(sec['label']))
                return

            # select median pupil datum from calibration list and use its detection method as mapping method
            mapping_method = '3d' if '3d' in calib_list[len(calib_list)//2]['method'] else '2d'

            map_slc = parse_range(sec['mapping_range'], len(self.g_pool.timestamps))
            map_list = list(chain(*self.g_pool.pupil_positions_by_frame[map_slc]))

            fake = setup_fake_pool(self.g_pool.capture.frame_size, mapping_method)
            generator_args = (fake, ref_list, calib_list, map_list)

            logger.info('Calibrating "{}"...'.format(sec['label']))
            self.bg_tasks[idx] = bh.Task_Proxy(sec['label'], calibrate_and_map, args=generator_args)
        self.persistent.save()

    def cleanup(self):
        if self.detection_proxy:
            self.detection_proxy.cancel()
        for proxy in self.bg_tasks:
            if proxy is not None:
                proxy.cancel()
        self.deinit_gui()
        self.persistent.save()
