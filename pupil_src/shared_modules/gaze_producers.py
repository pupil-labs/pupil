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
from time import time
from calibration_routines import gaze_mapping_plugins
from calibration_routines.finish_calibration import select_calibration_method
from file_methods import load_object, save_object

import background_helper as bh
import zmq_tools
from itertools import chain
import random

import logging
logger = logging.getLogger(__name__)

gaze_mapping_plugins_by_name = {p.__name__: p for p in gaze_mapping_plugins}


class Empty(object):
        pass


def setup_fake_pool(frame_size, detection_mode,rec_dir):
    cap = Empty()
    cap.frame_size = frame_size
    pool = Empty()
    pool.capture = cap
    pool.get_timestamp = time
    pool.detection_mapping_mode = detection_mode
    pool.rec_dir = rec_dir
    pool.app = 'player'
    return pool


random_colors = ((0.66015625, 0.859375, 0.4609375, 0.8),
                 (0.99609375, 0.84375, 0.3984375, 0.8),
                 (0.46875, 0.859375, 0.90625, 0.8),
                 (0.984375, 0.59375, 0.40234375, 0.8),
                 (0.66796875, 0.61328125, 0.9453125, 0.8),
                 (0.99609375, 0.37890625, 0.53125, 0.8))


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

    def get_init_dict(self):
        return {}


def calibrate_and_map(g_pool, ref_list, calib_list, map_list):
    yield "calibrating",[]
    method, result = select_calibration_method(g_pool, calib_list, ref_list)
    if result['subject'] != 'calibration.failed':
        logger.info('Offline calibration successful. Starting mapping using {}.'.format(method))
        name, args = result['name'], result['args']
        gaze_mapper_cls = gaze_mapping_plugins_by_name[name]
        gaze_mapper = gaze_mapper_cls(Empty(), **args)

        for idx, datum in enumerate(map_list):
            mapped_gaze = gaze_mapper.on_pupil_datum(datum)
            if mapped_gaze:
                progress = (100 * (idx+1)/len(map_list))
                if progress == 100:
                    progress = "Mapping complete."
                else:
                    progress = "Mapping..{}%".format(int(progress))
                yield progress, mapped_gaze
    else:
        yield "calibration failed",[]


def make_section_dict(calib_range, map_range):
        return {'calibration_range': calib_range,
                'mapping_range': map_range,
                'mapping_method': '3d',
                'calibration_method':"circle_marker",
                'status': 'unmapped',
                'color': random.choice(random_colors),
                'gaze_positions':[],
                'bg_task':None}


class Offline_Calibration(Gaze_Producer_Base):
    session_data_version = 3

    def __init__(self, g_pool,manual_ref_edit_mode=False):
        super().__init__(g_pool)
        self.manual_ref_edit_mode = manual_ref_edit_mode
        self.menu = None
        self.process_pipe = None


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
            session_data['circle_marker_positions'] = []
            session_data['manual_ref_positions'] = []
        self.sections = session_data['sections']
        self.circle_marker_positions = session_data['circle_marker_positions']
        self.manual_ref_positions = session_data['manual_ref_positions']
        if self.circle_marker_positions:
            self.detection_progress = 100.0
            for s in self.sections:
                self.calibrate_section(s)
            self.correlate_and_publish()
        else:
            self.detection_progress = 0.0
            self.start_detection_task()

    def append_section(self):
        map_range = [0, len(self.g_pool.timestamps)]
        calib_range = [len(self.g_pool.timestamps)//10, len(self.g_pool.timestamps)//2]
        sec = make_section_dict(calib_range,map_range)
        self.sections.append(sec)
        if self.menu is not None:
            self.append_section_menu(sec, collapsed=False)

    def start_detection_task(self):
        self.process_pipe = zmq_tools.Msg_Pair_Server(self.g_pool.zmq_ctx)
        self.circle_marker_positions = []
        source_path = self.g_pool.capture.source_path
        timestamps_path = os.path.join(self.g_pool.rec_dir, "world_timestamps.npy")
        self.notify_all({'subject': 'circle_detector_process.should_start',
                         'source_path': source_path, 'timestamps_path': timestamps_path,"pair_url":self.process_pipe.url})

    def init_gui(self):
        def clear_markers():
            self.manual_ref_positions = []
        self.menu = ui.Scrolling_Menu("Offline Calibration", pos=(-660, 20), size=(300, 500))
        self.g_pool.gui.append(self.menu)

        self.menu.append(ui.Info_Text('"Detection" searches for calibration markers in the world video.'))
        # self.menu.append(ui.Button('Redetect', self.start_detection_task))
        slider = ui.Slider('detection_progress', self, label='Detection Progress', setter=lambda _: _)
        slider.display_format = '%3.0f%%'
        self.menu.append(slider)
        self.menu.append(ui.Switch('manual_ref_edit_mode',self,label="Manual calibration edit mode."))
        self.menu.append(ui.Button('Clear manual markers',clear_markers))
        self.menu.append(ui.Button('Add section', self.append_section))

        for sec in self.sections:
            self.append_section_menu(sec)
        self.on_window_resize(glfwGetCurrentContext(), *glfwGetWindowSize(glfwGetCurrentContext()))

    def append_section_menu(self, sec, collapsed=True):
        section_menu = ui.Growing_Menu('Gaze Section')
        section_menu.collapsed = collapsed
        section_menu.color = RGBA(*sec['color'])

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

        section_menu.append(ui.Selector('calibration_method',sec,label="Calibration Method",selection=['circle_marker','natural_features'] ))
        section_menu.append(ui.Selector('mapping_method', sec, label='Calibration Mode',selection=['2d','3d']))
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
        return {'manual_ref_edit_mode':self.manual_ref_edit_mode}

    def on_notify(self, notification):
        subject = notification['subject']
        if subject == 'pupil_positions_changed':
            for s in self.sections:
                self.calibrate_section(s)

    def on_click(self, pos, button, action):
        if action == GLFW_PRESS and self.manual_ref_edit_mode:
            manual_refs_in_frame = [r for r in self.manual_ref_positions if self.g_pool.capture.get_frame_index() in r['index_range'] ]
            for ref in manual_refs_in_frame:
                if np.sqrt((pos[0]-ref['screen_pos'][0])**2 + (pos[1]-ref['screen_pos'][1])**2) < 15:  # img pixels
                    del self.manual_ref_positions[self.manual_ref_positions.index(ref)]
                    return
            new_ref = { 'screen_pos': pos,
                        'norm_pos': normalize(pos, self.g_pool.capture.frame_size, flip_y=True),
                        'index': self.g_pool.capture.get_frame_index(),
                        'index_range': tuple(range(self.g_pool.capture.get_frame_index()-5,self.g_pool.capture.get_frame_index()+5)),
                        'timestamp': self.g_pool.timestamps[self.g_pool.capture.get_frame_index()]
                        }
            self.manual_ref_positions.append(new_ref)

    def recent_events(self, events):

        if self.process_pipe and self.process_pipe.new_data:
            topic, msg = self.process_pipe.recv()
            if topic == 'progress':
                recent = msg.get('data', [])
                progress, data = zip(*recent)
                self.circle_marker_positions.extend([d for d in data if d])
                self.detection_progress = progress[-1]
            elif topic == 'finished':
                self.detection_progress = 100.
                self.process_pipe = None
                for s in self.sections:
                    self.calibrate_section(s)
            elif topic == 'exception':
                self.process_pipe = None
                self.detection_progress = 0.
                logger.info('Marker detection was interrupted')
                logger.debug('Reason: {}'.format(msg.get('reason', 'n/a')))


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
        if sec['bg_task']:
            sec['bg_task'].cancel()

        sec['status'] = 'starting calibration'#this will be overwritten on sucess
        sec['gaze_positions'] = []  # reset interim buffer for given section

        calib_list = list(chain(*self.g_pool.pupil_positions_by_frame[slice(*sec['calibration_range'])]))
        map_list = list(chain(*self.g_pool.pupil_positions_by_frame[slice(*sec['mapping_range'])]))

        if sec['calibration_method'] == 'circle_marker':
            ref_list = [r for r in self.circle_marker_positions if sec['calibration_range'][0] <= r['index'] <= sec['calibration_range'][1]]
        elif sec['calibration_method'] == 'natural_features':
            ref_list = self.manual_ref_positions
        if not calib_list:
            logger.error('No pupil data to calibrate section "{}"'.format(self.sections.index(sec) + 1))
            sec['status'] = 'calibration failed'
            return

        if not calib_list:
            logger.error('No referece marker data to calibrate section "{}"'.format(self.sections.index(sec) + 1))
            sec['status'] = 'calibration failed'
            return

        if sec["mapping_method"] == '3d' and '2d' in calib_list[len(calib_list)//2]['method']:
            # select median pupil datum from calibration list and use its detection method as mapping method
            logger.warning("Pupil data is 2d, calibration and mapping mode forced to 2d.")
            sec["mapping_method"] = '2d'

        fake = setup_fake_pool(self.g_pool.capture.frame_size, detection_mode=sec["mapping_method"],rec_dir=self.g_pool.rec_dir)
        generator_args = (fake, ref_list, calib_list, map_list)

        logger.info('Calibrating "{}" in {} mode...'.format(self.sections.index(sec) + 1,sec["mapping_method"]))
        sec['bg_task'] = bh.Task_Proxy('{}'.format(self.sections.index(sec) + 1), calibrate_and_map, args=generator_args)

    def gl_display(self):
        ref_point_norm = [r['norm_pos'] for r in self.circle_marker_positions if  self.g_pool.capture.get_frame_index() == r['index']]
        draw_points_norm(ref_point_norm,size=35, color=RGBA(0, .5, 0.5, .7))
        draw_points_norm(ref_point_norm,size=5, color=RGBA(.0, .9, 0.0, 1.0))

        manual_refs_in_frame = [r['norm_pos'] for r in self.manual_ref_positions if  self.g_pool.capture.get_frame_index() in r['index_range']]
        draw_points_norm(manual_refs_in_frame,size=35, color=RGBA(.0, .0, 0.9, .8))
        draw_points_norm(manual_refs_in_frame,size=5, color=RGBA(.0, .9, 0.0, 1.0))

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

        gl.glTranslatef(0, .04, 0)



        for s in self.sections:
            if s['calibration_method'] == "natural_features":
                draw_points([(m['index'],0) for m in self.manual_ref_positions],size=12,color=RGBA(.0, .0, 0.9, .8))
            else:
                draw_points([(m['index'],0) for m in self.circle_marker_positions],size=12,color=RGBA(0, .5, 0.5, .7))
            cal_slc = slice(*s['calibration_range'])
            map_slc = slice(*s['mapping_range'])
            color = RGBA(*s['color'])
            draw_polyline([(cal_slc.start, 0), (cal_slc.stop, 0)], color=color, line_type=gl.GL_LINES, thickness=8)
            draw_polyline([(map_slc.start, 0), (map_slc.stop, 0)], color=color, line_type=gl.GL_LINES, thickness=2)
            gl.glTranslatef(0, .04, 0)



        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()

    def on_window_resize(self, window, w, h):
        self.win_size = w, h

    def cleanup(self):
        if self.process_pipe:
            self.process_pipe.send(topic='terminate',payload={})
            self.process_pipe.socket.close()
            self.process_pipe = None
        for sec in self.sections:
            if sec['bg_task']:
                sec['bg_task'].cancel()
            sec['bg_task'] = None
            sec["gaze_positions"] = []

        session_data = {}
        session_data['sections'] = self.sections
        session_data['version'] = self.session_data_version
        session_data['manual_ref_positions'] = self.manual_ref_positions
        if self.detection_progress == 100.0:
            session_data['circle_marker_positions'] = self.circle_marker_positions
        else:
            session_data['circle_marker_positions'] = []
        save_object(session_data, os.path.join(self.result_dir, 'offline_calibration_gaze'))
        self.deinit_gui()
