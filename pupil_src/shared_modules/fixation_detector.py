'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

"""
Fixations general knowledge from literature review
    + Goldberg et al. - fixations rarely < 100ms and range between 200ms and 400ms in duration (Irwin, 1992 - fixations dependent on task between 150ms - 600ms)
    + Very short fixations are considered not meaningful for studying behavior - eye+brain require time for info to be registered (see Munn et al. APGV, 2008)
    + Fixations are rarely longer than 800ms in duration
        + Smooth Pursuit is exception and different motif
        + If we do not set a maximum duration, we will also detect smooth pursuit (which is acceptable since we compensate for VOR)
Terms
    + dispersion (spatial) = how much spatial movement is allowed within one fixation (in visual angular degrees or pixels)
    + duration (temporal) = what is the minimum time required for gaze data to be within dispersion threshold?
"""

import os
import csv
import numpy as np
import cv2

from bisect import bisect_left, bisect_right
from scipy.spatial.distance import pdist
from collections import deque
from itertools import chain
from pyglui import ui
from pyglui.cygl.utils import draw_circle, RGBA
from pyglui.pyfontstash import fontstash

from methods import denormalize
from player_methods import transparent_circle
from plugin import Analysis_Plugin_Base

import background_helper as bh

# logging
import logging
logger = logging.getLogger(__name__)


class Empty(object):
        pass


class Fixation_Detector_Base(Analysis_Plugin_Base):
    icon_chr = chr(0xec02)
    icon_font = 'pupil_icons'


def fixation_from_data(dispersion, method, base_data, timestamps=None):
    norm_pos = np.mean([gp['norm_pos'] for gp in base_data], axis=0).tolist()
    dispersion = np.rad2deg(dispersion)  # in degrees

    fix = {
        'topic': 'fixation',
        'norm_pos': norm_pos,
        'dispersion': dispersion,
        'method': method,
        'base_data': list(base_data),
        'timestamp': base_data[0]['timestamp'],
        'duration': (base_data[-1]['timestamp'] - base_data[0]['timestamp']) * 1000,
        'confidence': float(np.mean([gp['confidence'] for gp in base_data]))
    }
    if method == 'pupil':
        fix['gaze_point_3d'] = np.mean([gp['gaze_point_3d'] for gp in base_data
                                       if 'gaze_point_3d' in gp], axis=0).tolist()
    if timestamps is not None:
        start, end = base_data[0]['timestamp'], base_data[-1]['timestamp']
        start, end = np.searchsorted(timestamps, [start, end])
        end = min(end, len(timestamps) - 1)  # fix `list index out of range` error
        fix['start_frame_index'] = start
        fix['end_frame_index'] = end
        fix['mid_frame_index'] = (start + end) // 2
    return fix


def vector_dispersion(vectors):
    distances = pdist(vectors, metric='cosine')
    # use 20% biggest distances, but at least 4, see reasoning at
    # https://github.com/pupil-labs/pupil/issues/1133#issuecomment-382412175
    distances.sort()  # sort by distance
    cut_off = np.max([distances.shape[0] // 5, 4])
    return np.arccos(1. - distances[-cut_off:].mean())


def gaze_dispersion(capture, gaze_subset, use_pupil=True):
    if use_pupil:
        data = [[], []]
        # for each eye collect gaze positions that contain pp for the given eye
        data[0] = [gp for gp in gaze_subset if any(('3d' in pp['method'] and pp['id'] == 0)
                                                   for pp in gp['base_data'])]
        data[1] = [gp for gp in gaze_subset if any(('3d' in pp['method'] and pp['id'] == 1)
                                                   for pp in gp['base_data'])]

        method = 'pupil'
        # choose eye with more data points. alternatively data that spans longest time range
        eye_id = 1 if len(data[1]) > len(data[0]) else 0
        base_data = data[eye_id]

        all_pp = chain.from_iterable((gp['base_data'] for gp in base_data))
        pp_with_eye_id = (pp for pp in all_pp if pp['id'] == eye_id)
        vectors = np.array([pp['circle_3d']['normal'] for pp in pp_with_eye_id], dtype=np.float32)
    else:
        method = 'gaze'
        base_data = gaze_subset
        locations = np.array([gp['norm_pos'] for gp in gaze_subset])

        # denormalize
        width, height = capture.frame_size
        locations[:, 0] *= width
        locations[:, 1] = (1. - locations[:, 1]) * height

        # undistort onto 3d plane
        vectors = capture.intrinsics.unprojectPoints(locations)

    dist = vector_dispersion(vectors)
    return dist, method, base_data


def detect_fixations(capture, gaze_data, max_dispersion, min_duration, max_duration):
    yield "Detecting fixations...", []
    use_pupil = 'gaze_normal_3d' in gaze_data[0]
    logger.info('Starting fixation detection using {} data...'.format('3d' if use_pupil else '2d'))

    Q = deque()
    enum = deque(gaze_data)
    while enum:
        # check if Q contains enough data
        if len(Q) < 2 or Q[-1]['timestamp'] - Q[0]['timestamp'] < min_duration:
            datum = enum.popleft()
            Q.append(datum)
            continue

        # min duration reached, check for fixation
        dispersion, origin, base_data = gaze_dispersion(capture, Q, use_pupil=use_pupil)
        if dispersion > max_dispersion:
            # not a fixation, move forward
            Q.popleft()
            continue

        left_idx = len(Q)

        # minimal fixation found. collect maximal data
        # to perform binary search for fixation end
        while enum:
            datum = enum[0]
            if datum['timestamp'] > Q[0]['timestamp'] + max_duration:
                break  # maximum data found
            Q.append(enum.popleft())

        # check for fixation with maximum duration
        dispersion, origin, base_data = gaze_dispersion(capture, Q, use_pupil=use_pupil)
        if dispersion <= max_dispersion:
            yield 'Detecting fixations...', [fixation_from_data(dispersion, origin, base_data, capture.timestamps)]
            Q = deque()  # discard old Q
            continue

        slicable = list(Q)  # deque does not support slicing
        right_idx = len(Q)

        # binary search
        while left_idx + 1 < right_idx:
            middle_idx = (left_idx + right_idx) // 2 + 1
            dispersion, origin, base_data = gaze_dispersion(capture, slicable[:middle_idx], use_pupil=use_pupil)

            if dispersion <= max_dispersion:
                left_idx = middle_idx - 1
            else:
                right_idx = middle_idx - 1

        middle_idx = (left_idx + right_idx) // 2
        # if dispersion > max_dispersion:
        dispersion, origin, base_data = gaze_dispersion(capture, slicable[:middle_idx], use_pupil=use_pupil)

        # Create fixation datum
        fixation_datum = fixation_from_data(dispersion, origin, base_data, capture.timestamps)

        # Assert constraints
        assert dispersion <= max_dispersion, 'Fixation too big: {}'.format(fixation_datum)
        assert min_duration <= fixation_datum['duration'] / 1000, 'Fixation too short: {}'.format(fixation_datum)
        assert fixation_datum['duration'] / 1000 <= max_duration, 'Fixation too long: {}'.format(fixation_datum)

        yield 'Detecting fixations...', [fixation_datum]
        Q = deque()  # clear queue
        enum.extendleft(slicable[middle_idx:])

    yield "Fixation detection complete", []


class Offline_Fixation_Detector(Fixation_Detector_Base):
    '''Dispersion-duration-based fixation detector.

    This plugin detects fixations based on a dispersion threshold in terms of
    degrees of visual angle within a given duration window. It tries to maximize
    the length of classified fixations within the duration window, e.g. instead
    of creating two consecutive fixations of length 300 ms it creates a single
    fixation with length 600 ms. Fixations do not overlap. Binary search is used
    to find the correct fixation length within the duration window.

    If 3d pupil data is available the fixation dispersion will be calculated
    based on the positional angle of the eye. These fixations have their method
    field set to "pupil". If no 3d pupil data is available the plugin will
    assume that the gaze data is calibrated and calculate the dispersion in
    visual angle within the coordinate system of the world camera. These
    fixations will have their method field set to "gaze".
    '''
    def __init__(self, g_pool, max_dispersion=3.0, min_duration=300, max_duration=1000, show_fixations=True):
        super().__init__(g_pool)
        # g_pool.min_data_confidence
        self.max_dispersion = max_dispersion
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.show_fixations = show_fixations
        self.current_fixation_details = None
        self.fixations = deque()
        self.prev_index = -1
        self.bg_task = None
        self.status = ''
        self.notify_all({'subject': 'fixation_detector.should_recalculate', 'delay': .5})

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Fixation Detector'

        def set_max_dispersion(new_value):
            self.max_dispersion = new_value
            self.notify_all({'subject': 'fixation_detector.should_recalculate', 'delay': 1.})

        def set_min_duration(new_value):
            self.min_duration = min(new_value, self.max_duration)
            self.notify_all({'subject': 'fixation_detector.should_recalculate', 'delay': 1.})

        def set_max_duration(new_value):
            self.max_duration = max(new_value, self.min_duration)
            self.notify_all({'subject': 'fixation_detector.should_recalculate', 'delay': 1.})

        def jump_next_fixation(_):
            cur_idx = self.last_frame_idx
            all_idc = [f['mid_frame_index'] for f in self.g_pool.fixations]
            if not all_idc:
                logger.warning('No fixations available')
                return
            # wrap-around index
            tar_fix = bisect_right(all_idc, cur_idx) % len(all_idc)
            self.notify_all({'subject': 'seek_control.should_seek',
                             'index': int(self.g_pool.fixations[tar_fix]['mid_frame_index'])})

        def jump_prev_fixation(_):
            cur_idx = self.last_frame_idx
            all_idc = [f['mid_frame_index'] for f in self.g_pool.fixations]
            if not all_idc:
                logger.warning('No fixations available')
                return
            # wrap-around index
            tar_fix = (bisect_left(all_idc, cur_idx) - 1) % len(all_idc)
            self.notify_all({'subject': 'seek_control.should_seek',
                             'index': int(self.g_pool.fixations[tar_fix]['mid_frame_index'])})

        for help_block in self.__doc__.split('\n\n'):
            help_str = help_block.replace('\n', ' ').replace('  ', '').strip()
            self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Info_Text("Press the export button or type 'e' to start the export."))

        self.menu.append(ui.Slider('max_dispersion', self, min=0.01, step=0.1, max=5.,
                                   label='Maximum Dispersion [degrees]', setter=set_max_dispersion))
        self.menu.append(ui.Slider('min_duration', self, min=10, step=10, max=4000,
                                   label='Minimum Duration [milliseconds]', setter=set_min_duration))
        self.menu.append(ui.Slider('max_duration', self, min=10, step=10, max=4000,
                                   label='Maximum Duration [milliseconds]', setter=set_max_duration))
        self.menu.append(ui.Text_Input('status', self, label='Detection progress:', setter=lambda x: None))
        self.menu.append(ui.Switch('show_fixations', self, label='Show fixations'))
        self.current_fixation_details = ui.Info_Text('')
        self.menu.append(self.current_fixation_details)

        self.next_fix_button = ui.Thumb('jump_next_fixation', setter=jump_next_fixation,
                                   getter=lambda: False, label=chr(0xe044), hotkey='f',
                                   label_font='pupil_icons')
        self.next_fix_button.status_text = 'Next Fixation'
        self.g_pool.quickbar.append(self.next_fix_button)

        self.prev_fix_button = ui.Thumb('jump_prev_fixation', setter=jump_prev_fixation,
                                   getter=lambda: False, label=chr(0xe045), hotkey='F',
                                   label_font='pupil_icons')
        self.prev_fix_button.status_text = 'Previous Fixation'
        self.g_pool.quickbar.append(self.prev_fix_button)

    def deinit_ui(self):
        self.remove_menu()
        self.current_fixation_details = None
        self.g_pool.quickbar.remove(self.next_fix_button)
        self.next_fix_button = None

    def cleanup(self):
        if self.bg_task:
            self.bg_task.cancel()
            self.bg_task = None

    def get_init_dict(self):
        return {'max_dispersion': self.max_dispersion, 'min_duration': self.min_duration,
                'max_duration': self.max_duration, 'show_fixations': self.show_fixations}

    def on_notify(self, notification):
        if notification['subject'] == 'gaze_positions_changed':
            logger.info('Gaze postions changed. Recalculating.')
            self._classify()
        if notification['subject'] == 'min_data_confidence_changed':
            logger.info('Minimal data confidence changed. Recalculating.')
            self._classify()
        elif notification['subject'] == 'fixation_detector.should_recalculate':
            self._classify()
        elif notification['subject'] == "should_export":
            self.export_fixations(notification['range'], notification['export_dir'])

    def _classify(self):
        '''
        classify fixations
        '''
        if self.g_pool.app == 'exporter':
            return

        if self.bg_task:
            self.bg_task.cancel()

        gaze_data = [gp for gp in self.g_pool.gaze_positions if gp['confidence'] >= self.g_pool.min_data_confidence]
        if not gaze_data:
            logger.error('No gaze data available to find fixations')
            self.status = 'Fixation detection failed'
            return

        cap = Empty()
        cap.frame_size = self.g_pool.capture.frame_size
        cap.intrinsics = self.g_pool.capture.intrinsics
        cap.timestamps = self.g_pool.capture.timestamps
        generator_args = (cap, gaze_data, np.deg2rad(self.max_dispersion),
                          self.min_duration / 1000, self.max_duration / 1000)

        self.fixations = deque()
        self.bg_task = bh.Task_Proxy('Fixation detection', detect_fixations, args=generator_args)

    def recent_events(self, events):
        if self.bg_task:
            recent = [d for d in self.bg_task.fetch()]
            if recent:
                progress, data = zip(*recent)
                self.fixations.extend(chain.from_iterable(data))
                self.status = progress[-1]
                if self.fixations:
                    current = self.fixations[-1]['timestamp']
                    progress = (current - self.g_pool.timestamps[0]) /\
                               (self.g_pool.timestamps[-1] - self.g_pool.timestamps[0])
                    self.menu_icon.indicator_stop = progress
            if self.bg_task.completed:
                self.status = "{} fixations detected".format(len(self.fixations))
                self.correlate_and_publish()
                self.bg_task = None
                self.menu_icon.indicator_stop = 0.

        frame = events.get('frame')
        if not frame:
            return

        self.last_frame_idx = frame.index
        events['fixations'] = self.g_pool.fixations_by_frame[frame.index]
        if self.show_fixations:
            for f in self.g_pool.fixations_by_frame[frame.index]:
                x = int(f['norm_pos'][0] * frame.width)
                y = int((1. - f['norm_pos'][1]) * frame.height)
                transparent_circle(frame.img, (x, y), radius=25., color=(0., 1., 1., 1.), thickness=3)
                cv2.putText(frame.img, '{}'.format(f['id']), (x + 30, y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 150, 100))

        if self.current_fixation_details and self.prev_index != frame.index:
            info = ''
            for f in self.g_pool.fixations_by_frame[frame.index]:
                info += 'Current fixation, {} of {}\n'.format(f['id'], len(self.g_pool.fixations))
                info += '    Confidence: {:.2f}\n'.format(f['confidence'])
                info += '    Duration: {:.2f} milliseconds\n'.format(f['duration'])
                info += '    Dispersion: {:.3f} degrees\n'.format(f['dispersion'])
                info += '    Frame range: {}-{}\n'.format(f['start_frame_index'] + 1, f['end_frame_index'] + 1)
                info += '    2d gaze pos: x={:.3f}, y={:.3f}\n'.format(*f['norm_pos'])
                if 'gaze_point_3d' in f:
                    info += '    3d gaze pos: x={:.3f}, y={:.3f}, z={:.3f}\n'.format(*f['gaze_point_3d'])
                else:
                    info += '    3d gaze pos: N/A\n'
                if f['id'] > 1:
                    prev_f = self.g_pool.fixations[f['id'] - 2]
                    time_lapsed = f['timestamp'] - prev_f['timestamp'] + prev_f['duration'] / 1000
                    info += '    Time since prev. fixation: {:.2f} seconds\n'.format(time_lapsed)
                else:
                    info += '    Time since prev. fixation: N/A\n'

                if f['id'] < len(self.g_pool.fixations):
                    next_f = self.g_pool.fixations[f['id']]
                    time_lapsed = next_f['timestamp'] - f['timestamp'] + f['duration'] / 1000
                    info += '    Time to next fixation: {:.2f} seconds\n'.format(time_lapsed)
                else:
                    info += '    Time to next fixation: N/A\n'

            self.current_fixation_details.text = info
            self.prev_index = frame.index

    def correlate_and_publish(self):
        fixations = sorted(self.fixations, key=lambda f: f['timestamp'])
        for idx, f in enumerate(fixations):
            f['id'] = idx + 1
        self.g_pool.fixations = fixations
        # now lets bin fixations into frames. Fixations are allotted to the first frame they appear in.
        fixations_by_frame = [[] for x in self.g_pool.timestamps]
        for f in self.fixations:
            for idx in range(f['start_frame_index'], f['end_frame_index'] + 1):
                fixations_by_frame[idx].append(f)

        self.g_pool.fixations_by_frame = fixations_by_frame
        self.notify_all({'subject': 'fixations_changed', 'delay': 1})

    @classmethod
    def csv_representation_keys(self):
        return ('id', 'start_timestamp', 'duration', 'start_frame_index', 'end_frame_index',
                'norm_pos_x', 'norm_pos_y', 'dispersion', 'confidence', 'method',
                'gaze_point_3d_x', 'gaze_point_3d_y', 'gaze_point_3d_z', 'base_data')

    @classmethod
    def csv_representation_for_fixation(self, fixation):
        return (fixation['id'],
                fixation['timestamp'],
                fixation['duration'],
                fixation['start_frame_index'],
                fixation['end_frame_index'],
                fixation['norm_pos'][0],
                fixation['norm_pos'][1],
                fixation['dispersion'],
                fixation['confidence'],
                fixation['method'],
                *fixation.get('gaze_point_3d', [None] * 3),  # expanded, hence * at beginning
                " ".join(['{}'.format(gp['timestamp']) for gp in fixation['base_data']]))

    def export_fixations(self, export_range, export_dir):
        """
        between in and out mark

            fixation report:
                - fixation detection method and parameters
                - fixation count

            fixation list:
                id | start_timestamp | duration | start_frame_index | end_frame_index |
                norm_pos_x | norm_pos_y | dispersion | confidence | method |
                gaze_point_3d_x | gaze_point_3d_y | gaze_point_3d_z | base_data
        """
        if not self.fixations:
            logger.warning('No fixations in this recording nothing to export')
            return

        fixations_in_section = chain.from_iterable(self.g_pool.fixations_by_frame[slice(*export_range)])
        fixations_in_section = list(dict([(f['id'], f) for f in fixations_in_section]).values()) # remove duplicates
        fixations_in_section.sort(key=lambda f: f['id'])

        with open(os.path.join(export_dir,'fixations.csv'),'w',encoding='utf-8',newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(self.csv_representation_keys())
            for f in fixations_in_section:
                csv_writer.writerow(self.csv_representation_for_fixation(f))
            logger.info("Created 'fixations.csv' file.")

        with open(os.path.join(export_dir,'fixation_report.csv'),'w',encoding='utf-8',newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(('fixation classifier','Dispersion_Duration'))
            csv_writer.writerow(('max_dispersion','{:0.3f} deg'.format(self.max_dispersion)) )
            csv_writer.writerow(('min_duration','{:0.3f} sec'.format(self.min_duration)) )
            csv_writer.writerow((''))
            csv_writer.writerow(('fixation_count',len(fixations_in_section)))
            logger.info("Created 'fixation_report.csv' file.")


class Fixation_Detector(Fixation_Detector_Base):
    '''Dispersion-duration-based fixation detector.

    This plugin detects fixations based on a dispersion threshold in terms of
    degrees of visual angle with a minimal duration. It publishes the fixation
    as soon as it complies with the constraints (dispersion and duration). This
    might result in a series of overlapping fixations. These will have their id
    field set to the same value which can be used to merge overlapping fixations.

    If 3d pupil data is available the fixation dispersion will be calculated
    based on the positional angle of the eye. These fixations have their method
    field set to "pupil". If no 3d pupil data is available the plugin will
    assume that the gaze data is calibrated and calculate the dispersion in
    visual angle with in the coordinate system of the world camera. These
    fixations will have their method field set to "gaze".

    The Offline Fixation Detector yields fixations that do not overlap.
    '''
    order = .19

    def __init__(self, g_pool, max_dispersion=3.0, min_duration=300, confidence_threshold=0.75):
        super().__init__(g_pool)
        self.queue = deque()
        self.min_duration = min_duration
        self.max_dispersion = max_dispersion
        self.confidence_threshold = confidence_threshold
        self.id_counter = 0

    def recent_events(self, events):
        events['fixations'] = []
        gaze = events['gaze_positions']

        self.queue.extend((gp for gp in gaze if gp['confidence'] >= self.confidence_threshold))

        try:  # use newest gaze point to determine age threshold
            age_threshold = self.queue[-1]['timestamp'] - self.min_duration / 1000.
            while self.queue[1]['timestamp'] < age_threshold:
                self.queue.popleft()  # remove outdated gaze points
        except IndexError:
            pass

        gaze_3d = [gp for gp in self.queue if '3d' in gp['base_data'][0]['method']]
        use_pupil = len(gaze_3d) > 0.8 * len(self.queue)

        base_data = gaze_3d if use_pupil else self.queue

        if len(base_data) <= 2 or base_data[-1]['timestamp'] - base_data[0]['timestamp'] < self.min_duration / 1000.:
            self.recent_fixation = None
            return

        dispersion, origin, base_data = gaze_dispersion(self.g_pool.capture, base_data, use_pupil)

        if dispersion < np.deg2rad(self.max_dispersion):
            new_fixation = fixation_from_data(dispersion, origin, base_data)
            if self.recent_fixation:
                new_fixation['id'] = self.recent_fixation['id']
            else:
                new_fixation['id'] = self.id_counter
                self.id_counter += 1

            events['fixations'].append(new_fixation)
            self.recent_fixation = new_fixation
        else:
            self.recent_fixation = None

    def gl_display(self):
        if self.recent_fixation:
            fs = self.g_pool.capture.frame_size  # frame height
            pt = denormalize(self.recent_fixation['norm_pos'], fs, flip_y=True)
            draw_circle(pt, radius=48., stroke_width=10., color=RGBA(1., 1., 0., 1.))
            self.glfont.draw_text(pt[0] + 48., pt[1], str(self.recent_fixation['id']))

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Fixation Detector'

        for help_block in self.__doc__.split('\n\n'):
            help_str = help_block.replace('\n', ' ').replace('  ', '').strip()
            self.menu.append(ui.Info_Text(help_str))

        self.menu.append(ui.Slider('max_dispersion', self, min=0.01, step=0.1, max=5.,
                                   label='Maximum Dispersion [degrees]'))
        self.menu.append(ui.Slider('min_duration', self, min=10, step=10, max=4000,
                                   label='Minimum Duration [milliseconds]'))

        self.menu.append(ui.Slider('confidence_threshold', self, min=0.0, max=1.0, label='Confidence Threshold'))

        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans', ui.get_opensans_font_path())
        self.glfont.set_size(22)
        self.glfont.set_color_float((0.2, 0.5, 0.9, 1.0))

    def deinit_ui(self):
        self.remove_menu()
        self.glfont = None

    def get_init_dict(self):
        return {'max_dispersion': self.max_dispersion, 'min_duration': self.min_duration,
                'confidence_threshold': self.confidence_threshold}
