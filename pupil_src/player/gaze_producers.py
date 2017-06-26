'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os,platform
import cv2
import numpy as np
from pyglui import ui
from plugin import Producer_Plugin_Base
from player_methods import correlate_data
from methods import normalize
from video_capture import File_Source, EndofVideoFileError
from circle_detector import find_concetric_circles
import OpenGL.GL as gl
from pyglui.cygl.utils import *
from glfw import *

from calibration_routines import gaze_mapping_plugins
from calibration_routines.finish_calibration import select_calibration_method
from file_methods import load_object, save_object

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

    try:
        src = File_Source(Global_Container(), source_path, np.load(timestamps_path), timed_playback=False)
        frame = src.get_frame()
        logger.info('Starting calibration marker detection...')
        frame_count = src.get_frame_count()

        while True:
            progress = 100.*frame.index/frame_count

            markers = find_concetric_circles(frame.gray, min_ring_count=3)
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
                marker_gray = frame.gray[slice(*row_slice), slice(*col_slice)]
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
        logger.info('Offline calibration successful. Starting mapping using {}.'.format(method))
        name, args = result['name'], result['args']
        gaze_mapper_cls = gaze_mapping_plugins_by_name[name]
        gaze_mapper = gaze_mapper_cls(Global_Container(), **args)

        for idx, datum in enumerate(map_list):
            mapped_gaze = gaze_mapper.on_pupil_datum(datum)
            if mapped_gaze:
                progress = (100 * (idx+1)/len(map_list))
                if progress == 100:
                    progress = "Mapping complete."
                else:
                    progress = "Mapping..{}%".format(int(progress))
                yield progress, mapped_gaze


def make_section_dict(calib_range, map_range):
        return {'calibration_range': calib_range,
                'mapping_range': map_range,
                'mapping_method': 'unknown',
                'status': 'unmapped',
                'color': list(np.random.rand(3)),
                'gaze_positions':[],
                'bg_task':None}

class Offline_Calibration(Gaze_Producer_Base):
    session_data_version = 3

    def __init__(self, g_pool):
        # mapping_cls_name='Dummy_Gaze_Mapper', mapping_args={},
        super().__init__(g_pool)
        self.menu = None

        self.result_dir = os.path.join(g_pool.rec_dir, 'offline_data')
        os.makedirs(self.result_dir, exist_ok=True)
        try:
            session_data = load_object(os.path.join(self.result_dir, 'offline_calibration_gaze'))
            if session_data['version'] != self.session_data_version:
                logger.warning("Session data from old version. Will not use this.")
                assert False
        except Exception as e:
            map_range = [0, len(self.g_pool.timestamps)]
            calib_range = [len(self.g_pool.timestamps)//10, len(self.g_pool.timestamps)//2]
            session_data = {}
            session_data['sections'] = [make_section_dict(calib_range, map_range), ]
            session_data['ref_positions'] = []
        self.sections = session_data['sections']
        self.ref_positions = session_data['ref_positions']

        self.detection_proxy = None
        if self.ref_positions:
            self.detection_progress = 100.0
            for s in self.sections:
                self.calibrate_section(s)
            self.correlate_and_publish()
        else:
            self.detection_progress = 0.0
            self.start_detection_task(None)

    def append_section(self):
        map_range = [0, len(self.g_pool.timestamps)]
        calib_range = [len(self.g_pool.timestamps)//10, len(self.g_pool.timestamps)//2]
        sec = make_section_dict(calib_range,map_range)
        self.sections.append(sec)
        if self.menu is not None:
            self.append_section_menu(sec, collapsed=False)

    def start_detection_task(self, _):
        # cancel current detection if running
        if self.detection_proxy is not None:
            self.detection_proxy.cancel()

        self.ref_positions = []
        source_path = self.g_pool.capture.source_path
        timestamps_path = os.path.join(self.g_pool.rec_dir, "world_timestamps.npy")

        self.detection_proxy = bh.Task_Proxy('Calibration Marker Detection',
                                             detect_marker_positions,
                                            force_spawn=platform.system() == 'Darwin',
                                             args=(source_path, timestamps_path))


    def init_gui(self):
        self.menu = ui.Scrolling_Menu("Offline Calibration", pos=(-660, 20), size=(300, 500))
        self.g_pool.gui.append(self.menu)

        self.menu.append(ui.Info_Text('"Detection" searches for calibration markers in the world video.'))
        self.menu.append(ui.Button('Redetect', self.start_detection_task))
        slider = ui.Slider('detection_progress', self, label='Detection Progress', setter=lambda _: _)
        slider.display_format = '%3.0f%%'
        self.menu.append(slider)

        self.menu.append(ui.Button('Add section', self.append_section))

        for sec in self.sections:
            self.append_section_menu(sec)
        self.on_window_resize(glfwGetCurrentContext(), *glfwGetWindowSize(glfwGetCurrentContext()))

    def append_section_menu(self, sec, collapsed=True):
        section_menu = ui.Growing_Menu('Gaze Section {}'.format(self.sections.index(sec) + 1))
        section_menu.collapsed = collapsed
        section_menu.color = RGBA(*sec['color'], 1.)

        def make_validate_fn(sec, key):
            def validate(input_obj):
                try:

                    assert type(input_obj) in (tuple,list)
                    assert type(input_obj[0]) is int
                    assert type(input_obj[1]) is int
                    assert 0 <= input_obj[0] <= input_obj[1] <=len(self.g_pool.timestamps)
                except:
                    pass
                else:
                    sec[key] = input_obj
            return validate

        def make_calibrate_fn(sec):
            def calibrate():
                self.calibrate_section(sec)
            return calibrate

        def make_remove_fn(sec):
            def remove():
                del self.menu[self.sections.index(sec)-len(self.sections)]
                del self.sections[self.sections.index(sec)]
                self.correlate_and_publish()

            return remove
        section_menu.append(ui.Text_Input('mapping_method', sec, label='Dection and mapping mode'))
        section_menu[-1].read_only = True
        section_menu.append(ui.Text_Input('status', sec, label='Calbiration Status', setter=lambda _: _))
        section_menu[-1].read_only = True
        section_menu.append(ui.Text_Input('calibration_range', sec, label='Calibration range',
                                          setter=make_validate_fn(sec, 'calibration_range')))
        section_menu.append(ui.Text_Input('mapping_range', sec, label='Mapping range',
                                          setter=make_validate_fn(sec, 'mapping_range')))
        section_menu.append(ui.Button('Recalibrate', make_calibrate_fn(sec)))
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
        if subject == 'pupil_positions_changed':
            # do not calibrate while detection task is still running
            if len(self.ref_positions) == len(self.g_pool.timestamps):
                for s in self.sections:
                    self.calibrate_section(s)

    def recent_events(self, events):
        if self.detection_proxy:
            recent = [d for d in self.detection_proxy.fetch()]

            if recent:
                progress, data = zip(*recent)
                self.ref_positions.extend(data)
                self.detection_progress = progress[-1]

            if self.detection_proxy.completed:
                self.detection_progress = 100.
                self.detection_proxy = None
                for s in self.sections:
                    self.calibrate_section(s)

        for sec in self.sections:
            if sec["bg_task"]:
                recent  = [d for d in sec["bg_task"].fetch()]
                if recent:
                    progress, data = zip(*recent)
                    sec['gaze_positions'].extend(chain(*data))
                    sec['status'] = progress[-1]
                if sec["bg_task"].completed:
                    self.correlate_and_publish()
                    sec['bg_task'] = None

    def correlate_and_publish(self):
        all_gaze = list(chain(*[s['gaze_positions'] for s in self.sections]))
        self.g_pool.gaze_positions = sorted(all_gaze, key=lambda d: d['timestamp'])
        self.g_pool.gaze_positions_by_frame = correlate_data(self.g_pool.gaze_positions, self.g_pool.timestamps)
        self.notify_all({'subject': 'gaze_positions_changed','delay':1})

    def calibrate_section(self,sec):
        if not self.ref_positions:
            logger.error('No markers have been found. Cannot calibrate.')
            return

        if sec['bg_task']:
            sec['bg_task'].cancel()

        sec['status'] = 'failed to calibrate'#this will be overwritten on sucess
        sec['gaze_positions'] = []  # reset interim buffer for given section

        calib_list = list(chain(*self.g_pool.pupil_positions_by_frame[slice(*sec['calibration_range'])]))
        ref_list = self.ref_positions[slice(*sec['calibration_range'])]
        ref_list = [r for r in ref_list if r is not None]
        if not calib_list:
            logger.error('There is not enough pupil data to calibrate section "{}"'.format(self.sections.index(sec) + 1))
            return

        if not calib_list:
            logger.error('There is not enough referece marker data to calibrate section "{}"'.format(self.sections.index(sec) + 1))
            return

        # select median pupil datum from calibration list and use its detection method as mapping method
        sec["mapping_method"] = '3d' if '3d' in calib_list[len(calib_list)//2]['method'] else '2d'

        map_list = list(chain(*self.g_pool.pupil_positions_by_frame[slice(*sec['mapping_range'])]))

        fake = setup_fake_pool(self.g_pool.capture.frame_size, sec["mapping_method"])
        generator_args = (fake, ref_list, calib_list, map_list)

        logger.info('Calibrating "{}"...'.format(self.sections.index(sec) + 1))
        sec['bg_task'] = bh.Task_Proxy('{}'.format(self.sections.index(sec) + 1), calibrate_and_map, args=generator_args)

    def gl_display(self):
        if len(self.ref_positions) > self.g_pool.capture.get_frame_index() and self.ref_positions[self.g_pool.capture.get_frame_index()]:
            ref_point_norm = self.ref_positions[self.g_pool.capture.get_frame_index()]['norm_pos']
            draw_points_norm((ref_point_norm,), color=RGBA(0, .5, 0.5, .7))
        padding = 30.
        max_ts = len(self.g_pool.timestamps)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        width, height = self.win_size
        h_pad = padding * (max_ts-2)/float(width)
        v_pad = padding * 1./(height-2)
        # ranging from 0 to len(timestamps)-1 (horizontal) and 0 to 1 (vertical)
        gl.glOrtho(-h_pad,  (max_ts-1)+h_pad, -v_pad, 1+v_pad, -1, 1)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        gl.glTranslatef(0, .03, 0)
        for s in self.sections:
            cal_slc = slice(*s['calibration_range'])
            map_slc = slice(*s['mapping_range'])
            color = RGBA(*s['color'], .8)
            draw_polyline([(cal_slc.start, 0), (cal_slc.stop, 0)], color=color, line_type=gl.GL_LINES, thickness=4)
            draw_polyline([(map_slc.start, 0), (map_slc.stop, 0)], color=color, line_type=gl.GL_LINES, thickness=2)
            gl.glTranslatef(0, .015, 0)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()

    def on_window_resize(self, window, w, h):
        self.win_size = w, h

    def cleanup(self):
        if self.detection_proxy:
            self.detection_proxy.cancel()
        for sec in self.sections:
            sec['bg_task'] = None
            sec["gaze_positions"] = None

        session_data = {}
        session_data['sections'] = self.sections
        session_data['version'] = self.session_data_version
        if len(self.ref_positions) == len(self.g_pool.timestamps):
            session_data['ref_positions'] = self.ref_positions
        else:
            session_data['ref_positions'] = []
        save_object(session_data,os.path.join(self.result_dir, 'offline_calibration_gaze'))
        self.deinit_gui()
