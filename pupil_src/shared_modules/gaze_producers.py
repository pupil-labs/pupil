'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import collections
import logging
import os
from itertools import chain, cycle
from time import time

import msgpack
import numpy as np
import OpenGL.GL as gl
import pyglui.cygl.utils as cygl_utils
# from copy import deepcopy
from pyglui import ui
from pyglui.pyfontstash import fontstash

import background_helper as bh
import file_methods as fm
import gl_utils
import glfw
import player_methods as pm
import zmq_tools
from calibration_routines import gaze_mapping_plugins
from calibration_routines.finish_calibration import select_calibration_method
from methods import normalize
from plugin import Producer_Plugin_Base

logger = logging.getLogger(__name__)

gaze_mapping_plugins_by_name = {p.__name__: p for p in gaze_mapping_plugins}


class Empty(object):
    pass


def setup_fake_pool(frame_size, intrinsics, detection_mode, rec_dir, min_calibration_confidence):
    cap = Empty()
    cap.frame_size = frame_size
    cap.intrinsics = intrinsics
    pool = Empty()
    pool.capture = cap
    pool.get_timestamp = time
    pool.detection_mapping_mode = detection_mode
    pool.min_calibration_confidence = min_calibration_confidence
    pool.rec_dir = rec_dir
    pool.app = 'player'
    return pool


colors = cycle(((0.66015625, 0.859375, 0.4609375, 0.8),
                (0.99609375, 0.84375, 0.3984375, 0.8),
                (0.46875, 0.859375, 0.90625, 0.8),
                (0.984375, 0.59375, 0.40234375, 0.8),
                (0.66796875, 0.61328125, 0.9453125, 0.8),
                (0.99609375, 0.37890625, 0.53125, 0.8)))


class Gaze_Producer_Base(Producer_Plugin_Base):
    uniqueness = 'by_base_class'
    order = .02
    icon_chr = chr(0xec14)
    icon_font = 'pupil_icons'

    def init_ui(self):
        self.add_menu()

        gaze_producer_plugins = [p for p in self.g_pool.plugin_by_name.values() if issubclass(p, Gaze_Producer_Base)]
        gaze_producer_plugins.sort(key=lambda p: p.__name__)

        self.menu_icon.order = 0.3

        def open_plugin(p):
            self.notify_all({'subject': 'start_plugin', 'name': p.__name__})

        # We add the capture selection menu
        self.menu.append(ui.Selector(
                                'gaze_producer',
                                setter=open_plugin,
                                getter=lambda: self.__class__,
                                selection=gaze_producer_plugins,
                                labels=[p.__name__.replace('_', ' ') for p in gaze_producer_plugins],
                                label='Gaze Producers'
                            ))

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        if 'frame' in events:
            frm_idx = events['frame'].index
            window = pm.enclosing_window(self.g_pool.timestamps, frm_idx)
            events['gaze'] = self.g_pool.gaze_positions.by_ts_window(window)


class Gaze_From_Recording(Gaze_Producer_Base):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.result_dir = os.path.join(g_pool.rec_dir, 'offline_data')
        os.makedirs(self.result_dir, exist_ok=True)
        try:
            session_data = fm.load_object(os.path.join(self.result_dir, 'manual_gaze_correction'))
        except OSError:
            session_data = {'dx': 0., 'dy': 0.}
        self.x_offset = session_data['dx']
        self.y_offset = session_data['dy']
        self.load_data_with_offset()

    def load_data_with_offset(self):
        gaze = fm.load_pldata_file(self.g_pool.rec_dir, 'gaze')
        self.g_pool.gaze_positions = pm.Bisector(gaze.data, gaze.timestamps)

        # self.g_pool.gaze_positions = deepcopy(self.g_pool.pupil_data['gaze'])
        # for gp in self.g_pool.gaze_positions:
        #     gp['norm_pos'][0] += self.x_offset
        #     gp['norm_pos'][1] += self.y_offset
        self.notify_all({'subject': 'gaze_positions_changed'})
        logger.debug('gaze positions changed')

    # def _set_offset_x(self, offset_x):
    #     self.x_offset = offset_x
    #     self.notify_all({'subject': 'manual_gaze_correction.offset_changed', 'delay': 1.})

    # def _set_offset_y(self, offset_y):
    #     self.y_offset = offset_y
    #     self.notify_all({'subject': 'manual_gaze_correction.offset_changed', 'delay': 1.})

    # def on_notify(self, notification):
    #     if notification['subject'] == 'manual_gaze_correction.offset_changed':
    #         self.load_data_with_offset()

    def init_ui(self):
        super().init_ui()
        self.menu.label = "Gaze Data  From Recording"
        self.menu.append(ui.Info_Text('Currently, gaze positions are loaded from the recording.'))
        # offset_menu = ui.Growing_Menu('Manual Correction')
        # offset_menu.append(ui.Info_Text('The manual correction feature allows you to apply' +
        #                                 ' a fixed offset to your gaze data.'))
        # offset_menu.append(ui.Slider('x_offset', self, min=-.5, step=0.01,
        #                              max=.5, setter=self._set_offset_x))
        # offset_menu.append(ui.Slider('y_offset', self, min=-.5, step=0.01,
        #                              max=.5, setter=self._set_offset_y))
        # offset_menu.collapsed = True
        # self.menu.append(offset_menu)

    def deinit_ui(self):
        super().deinit_ui()

    def cleanup(self):
        session_data = {'dx': self.x_offset, 'dy': self.y_offset, 'version': 0}
        fm.save_object(session_data, os.path.join(self.result_dir, 'manual_gaze_correction'))


def calibrate_and_map(g_pool, ref_list, calib_list, map_list, x_offset, y_offset):
    yield "calibrating", []
    calib_list = [msgpack.unpackb(serialized, raw=False, use_list=False)
                  for serialized in calib_list]
    method, result = select_calibration_method(g_pool, calib_list, ref_list)
    if result['subject'] != 'calibration.failed':
        logger.info('Offline calibration successful. Starting mapping using {}.'.format(method))
        name, args = result['name'], result['args']
        gaze_mapper_cls = gaze_mapping_plugins_by_name[name]
        gaze_mapper = gaze_mapper_cls(g_pool, **args)

        for idx_incoming, serialized in enumerate(map_list):
            datum = msgpack.unpackb(serialized, raw=False, use_list=False)
            mapped_gaze = gaze_mapper.on_pupil_datum(datum)

            # apply manual correction
            for idx_outgoing, gaze_datum in enumerate(mapped_gaze):
                # gp['norm_pos'] is a tuple by default
                gaze_norm_pos = list(gaze_datum['norm_pos'])
                gaze_norm_pos[1] += y_offset
                gaze_norm_pos[0] += x_offset
                gaze_datum['norm_pos'] = gaze_norm_pos
                serialized = msgpack.packb(gaze_datum, use_bin_type=True)
                mapped_gaze[idx_outgoing] = gaze_datum['timestamp'], serialized

            if mapped_gaze:
                progress = (100 * (idx_incoming + 1) / len(map_list))
                progress = "Mapping..{}%".format(int(progress))
                yield progress, mapped_gaze
        progress = "Mapping complete."
        yield progress, []
    else:
        # logger does not work here, because we are in a subprocess
        fail_message = 'Calibration failed: {}'.format(result['reason'])
        print(fail_message, len(calib_list), len(ref_list))
        yield fail_message, []


def make_section_dict(calib_range, map_range):
        return {'uid': np.random.rand(),  # ensures unique entry in self.sections
                'label': 'Unnamed section',
                'calibration_range': calib_range,
                'mapping_range': map_range,
                'mapping_method': '3d',
                'calibration_method': "circle_marker",
                'status': 'unmapped',
                'color': next(colors),
                'x_offset': 0.,
                'y_offset': 0.}


class Offline_Calibration(Gaze_Producer_Base):
    session_data_version = 9

    def __init__(self, g_pool, manual_ref_edit_mode=False):
        super().__init__(g_pool)
        self.timeline_line_height = 16
        self.manual_ref_edit_mode = manual_ref_edit_mode
        self.menu = None
        self.process_pipe = None

        self.result_dir = os.path.join(g_pool.rec_dir, 'offline_data')
        os.makedirs(self.result_dir, exist_ok=True)
        try:
            session_data = fm.load_object(os.path.join(self.result_dir, 'offline_calibration_gaze'))
            if session_data['version'] != self.session_data_version:
                logger.warning("Session data from old version. Will not use this.")
                assert False
        except (AssertionError, FileNotFoundError):
            session_data = {}
            max_idx = len(self.g_pool.timestamps) - 1
            session_data['sections'] = [make_section_dict((0, max_idx), (0, max_idx))]
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

    def append_section(self):
        max_idx = len(self.g_pool.timestamps) - 1
        sec = make_section_dict((0, max_idx), (0, max_idx))
        self.sections.append(sec)
        if self.menu is not None:
            self.append_section_menu(sec)

    def init_ui(self):
        super().init_ui()
        self.menu.label = "Offline Calibration"

        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans', ui.get_opensans_font_path())
        self.glfont.set_color_float((1., 1., 1., 1.))
        self.glfont.set_align_string(v_align='right', h_align='top')

        def use_as_natural_features():
            self.manual_ref_positions.extend(self.circle_marker_positions)
            self.manual_ref_positions.sort(key=lambda mr: mr['index'])

        def jump_next_natural_feature():
            self.manual_ref_positions.sort(key=lambda mr: mr['index'])
            current = self.g_pool.capture.get_frame_index()
            for nf in self.manual_ref_positions:
                if nf['index'] > current:
                    self.notify_all({'subject': 'seek_control.should_seek',
                                     'index': nf['index']})
                    return
            logger.error('No further natural feature available')

        def clear_natural_features():
            self.manual_ref_positions = []

        self.menu.append(ui.Info_Text('"Detection" searches for circle markers in the world video.'))
        # self.menu.append(ui.Button('Redetect', self.start_marker_detection))
        slider = ui.Slider('detection_progress', self, label='Detection Progress', setter=lambda _: _)
        slider.display_format = '%3.0f%%'
        self.menu.append(slider)

        toggle_label = 'Cancel circle marker detection' if self.process_pipe else 'Start circle marker detection'
        self.toggle_detection_button = ui.Button(toggle_label, self.toggle_marker_detection)
        self.menu.append(self.toggle_detection_button)

        self.menu.append(ui.Separator())

        self.menu.append(ui.Button('Use calibration markers as natural features', use_as_natural_features))
        self.menu.append(ui.Button('Jump to next natural feature', jump_next_natural_feature))
        self.menu.append(ui.Switch('manual_ref_edit_mode', self, label="Natural feature edit mode"))
        self.menu.append(ui.Button('Clear natural features', clear_natural_features))

        self.menu.append(ui.Info_Text('Calibration only considers pupil data that has an equal or higher confidence than the minimum calibration confidence.'))
        self.menu.append(ui.Slider('min_calibration_confidence', self.g_pool,
                                   step=.01, min=0.0, max=1.0,
                                   label='Minimum calibration confidence'))

        self.menu.append(ui.Button('Add section', self.append_section))

        # set to minimum height
        self.timeline = ui.Timeline('Calibration Sections', self.draw_sections, self.draw_labels, 1)
        self.g_pool.user_timelines.append(self.timeline)

        for sec in self.sections:
            self.append_section_menu(sec)
        self.on_window_resize(glfw.glfwGetCurrentContext(), *glfw.glfwGetWindowSize(glfw.glfwGetCurrentContext()))

    def deinit_ui(self):
        # needs to be called here since makes calls to the ui:
        self.cancel_marker_detection()
        super().deinit_ui()
        self.g_pool.user_timelines.remove(self.timeline)
        self.timeline = None
        self.glfont = None

    def append_section_menu(self, sec):
        section_menu = ui.Growing_Menu('Section Settings')
        section_menu.color = cygl_utils.RGBA(*sec['color'])

        def make_calibrate_fn(sec):
            def calibrate():
                self.calibrate_section(sec)
            return calibrate

        def make_remove_fn(sec):
            def remove():
                del self.menu[self.sections.index(sec) - len(self.sections)]
                del self.sections[self.sections.index(sec)]
                self.correlate_and_publish()

            return remove

        def set_trim_fn(button, sec, key):
            def trim(format_only=False):
                if format_only:
                    left_idx, right_idx = sec[key]
                else:
                    right_idx = self.g_pool.seek_control.trim_right
                    left_idx = self.g_pool.seek_control.trim_left
                    sec[key] = left_idx, right_idx

                time_fmt = key.replace('_', ' ').split(' ')[0].title() + ': '
                min_ts = self.g_pool.timestamps[0]
                for idx in (left_idx, right_idx):
                    ts = self.g_pool.timestamps[idx] - min_ts
                    minutes = ts // 60
                    seconds = ts - (minutes * 60.)
                    time_fmt += ' {:02.0f}:{:02.0f} -'.format(abs(minutes), seconds)
                button.outer_label = time_fmt[:-2]  # remove final ' -'
            button.function = trim

        section_menu.append(ui.Text_Input('label', sec, label='Label'))
        section_menu.append(ui.Selector('calibration_method', sec,
                                        label="Calibration Method",
                                        labels=['Circle Marker', 'Natural Features'],
                                        selection=['circle_marker', 'natural_features']))
        section_menu.append(ui.Selector('mapping_method', sec, label='Calibration Mode',selection=['2d', '3d']))
        section_menu.append(ui.Text_Input('status', sec, label='Calibration Status', setter=lambda _: _))

        section_menu.append(ui.Info_Text('This section is calibrated using reference markers found in a user set range "Calibration". The calibration is used to map pupil to gaze positions within a user set range "Mapping". Drag trim marks in the timeline to set a range and apply it.'))

        calib_range_button = ui.Button('Set from trim marks', None)
        set_trim_fn(calib_range_button, sec, 'calibration_range')
        calib_range_button.function(format_only=True)  # set initial label
        section_menu.append(calib_range_button)

        mapping_range_button = ui.Button('Set from trim marks', None)
        set_trim_fn(mapping_range_button, sec, 'mapping_range')
        mapping_range_button.function(format_only=True)  # set initial label
        section_menu.append(mapping_range_button)

        section_menu.append(ui.Button('Recalibrate', make_calibrate_fn(sec)))
        section_menu.append(ui.Button('Remove section', make_remove_fn(sec)))

        # manual gaze correction menu
        offset_menu = ui.Growing_Menu('Manual Correction')
        offset_menu.append(ui.Info_Text('The manual correction feature allows you to apply' +
                                        ' a fixed offset to your gaze data.'))
        offset_menu.append(ui.Slider('x_offset', sec, min=-.5, step=0.01, max=.5))
        offset_menu.append(ui.Slider('y_offset', sec, min=-.5, step=0.01, max=.5))
        offset_menu.collapsed = True
        section_menu.append(offset_menu)
        self.menu.append(section_menu)

    def get_init_dict(self):
        return {'manual_ref_edit_mode': self.manual_ref_edit_mode}

    def on_notify(self, notification):
        subject = notification['subject']
        if subject == 'pupil_positions_changed':
            for s in self.sections:
                self.calibrate_section(s)
        elif subject == 'gaze_positions_changed':
            self.save_offline_data()

    def on_click(self, pos, button, action):
        if action == glfw.GLFW_PRESS and self.manual_ref_edit_mode:
            manual_refs_in_frame = [r for r in self.manual_ref_positions
                                    if self.g_pool.capture.get_frame_index() == r['index']]
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
        super().recent_events(events)

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
                logger.warning('Calibration marker detection raised exception:\n{}'.format(msg['reason']))
                self.process_pipe = None
                self.detection_progress = 0.
                logger.info('Marker detection was interrupted')
                logger.debug('Reason: {}'.format(msg.get('reason', 'n/a')))
            self.menu_icon.indicator_stop = self.detection_progress / 100.

        for sec in self.sections:
            if 'bg_task' in sec:
                for progress, gaze_data in sec["bg_task"].fetch():
                    for timestamp, serialized in gaze_data:
                        gaze_datum = fm.Serialized_Dict(msgpack_bytes=serialized)
                        sec['gaze'].append(gaze_datum)
                        sec['gaze_ts'].append(timestamp)
                    sec['status'] = progress
                if sec["bg_task"].completed:
                    self.correlate_and_publish()
                    del sec['bg_task']

    def correlate_and_publish(self):
        gaze_data = list(chain.from_iterable((s['gaze'] for s in self.sections)))
        gaze_ts = list(chain.from_iterable((s['gaze_ts'] for s in self.sections)))
        self.g_pool.gaze_positions = pm.Bisector(gaze_data, gaze_ts)
        self.notify_all({'subject': 'gaze_positions_changed', 'delay': 1})

    def calibrate_section(self, sec):
        if 'bg_task' in sec:
            sec['bg_task'].cancel()

        sec['status'] = 'Starting calibration' # This will be overwritten on success

        try:
            sec['gaze'].clear()
            sec['gaze_ts'].clear()
        except KeyError:
            sec['gaze'] = collections.deque()
            sec['gaze_ts'] = collections.deque()

        calibration_window = pm.exact_window(self.g_pool.timestamps, sec['calibration_range'])
        mapping_window = pm.exact_window(self.g_pool.timestamps, sec['mapping_range'])

        calibration_pupil_pos = self.g_pool.pupil_positions.by_ts_window(calibration_window)
        mapping_pupil_pos = self.g_pool.pupil_positions.by_ts_window(mapping_window)

        if sec['calibration_method'] == 'circle_marker':
            ref_list = self.circle_marker_positions
        elif sec['calibration_method'] == 'natural_features':
            ref_list = self.manual_ref_positions

        start = sec['calibration_range'][0]
        end = sec['calibration_range'][1]
        ref_list = [r for r in ref_list if start <= r['index'] <= end]

        if not len(calibration_pupil_pos):
            logger.error('No pupil data to calibrate section "{}"'.format(self.sections.index(sec) + 1))
            sec['status'] = 'Calibration failed. Not enough pupil positions.'
            return

        if not ref_list:
            logger.error('No referece marker data to calibrate section "{}"'.format(self.sections.index(sec) + 1))
            sec['status'] = 'Calibration failed. Not enough reference positions.'
            return

        if sec["mapping_method"] == '3d' and '2d' in calibration_pupil_pos[len(calibration_pupil_pos)//2]['method']:
            # select median pupil datum from calibration list and use its detection method as mapping method
            logger.warning("Pupil data is 2d, calibration and mapping mode forced to 2d.")
            sec["mapping_method"] = '2d'

        fake = setup_fake_pool(self.g_pool.capture.frame_size,
                               self.g_pool.capture.intrinsics,
                               sec["mapping_method"],
                               self.g_pool.rec_dir,
                               self.g_pool.min_calibration_confidence)

        calibration_pupil_pos = [pp.serialized for pp in calibration_pupil_pos]
        mapping_pupil_pos = [pp.serialized for pp in mapping_pupil_pos]

        generator_args = (fake, ref_list, calibration_pupil_pos, mapping_pupil_pos, sec['x_offset'], sec['y_offset'])

        logger.info('Calibrating section {} ({}) in {} mode...'.format(self.sections.index(sec) + 1, sec['label'], sec["mapping_method"]))
        sec['bg_task'] = bh.Task_Proxy('{}'.format(self.sections.index(sec) + 1), calibrate_and_map, args=generator_args)

    def gl_display(self):
        # normalize coordinate system, no need this step in utility functions
        with gl_utils.Coord_System(0, 1, 0, 1):
            ref_point_norm = [r['norm_pos'] for r in self.circle_marker_positions
                              if self.g_pool.capture.get_frame_index() == r['index']]
            cygl_utils.draw_points(ref_point_norm, size=35, color=cygl_utils.RGBA(0, .5, 0.5, .7))
            cygl_utils.draw_points(ref_point_norm, size=5, color=cygl_utils.RGBA(.0, .9, 0.0, 1.0))

            manual_refs_in_frame = [r for r in self.manual_ref_positions
                                    if self.g_pool.capture.get_frame_index() in r['index_range']]
            current = self.g_pool.capture.get_frame_index()
            for mr in manual_refs_in_frame:
                if mr['index'] == current:
                    cygl_utils.draw_points([mr['norm_pos']], size=35, color=cygl_utils.RGBA(.0, .0, 0.9, .8))
                    cygl_utils.draw_points([mr['norm_pos']], size=5, color=cygl_utils.RGBA(.0, .9, 0.0, 1.0))
                else:
                    distance = abs(current - mr['index'])
                    range_radius = (mr['index_range'][-1] - mr['index_range'][0]) // 2
                    # scale alpha [.1, .9] depending on distance to current frame
                    alpha = distance / range_radius
                    alpha = 0.1 * alpha + 0.9 * (1. - alpha)
                    # Use draw_progress instead of draw_circle. draw_circle breaks
                    # because of the normalized coord-system.
                    cygl_utils.draw_progress(mr['norm_pos'], 0., 0.999,
                                             inner_radius=20.,
                                             outer_radius=35.,
                                             color=cygl_utils.RGBA(.0, .0, 0.9, alpha))
                    cygl_utils.draw_points([mr['norm_pos']], size=5, color=cygl_utils.RGBA(.0, .9, 0.0, alpha))

        # calculate correct timeline height. Triggers timeline redraw only if changed
        self.timeline.content_height = max(0.001, self.timeline_line_height * len(self.sections))

    def draw_sections(self, width, height, scale):
        t0, t1 = self.g_pool.timestamps[0], self.g_pool.timestamps[-1]
        pixel_to_time_fac = height / (t1 - t0)
        with gl_utils.Coord_System(t0, t1, height, 0):
            gl.glTranslatef(0, 0.001 + scale * self.timeline_line_height / 2, 0)
            for s in self.sections:
                cal_slc = slice(*s['calibration_range'])
                map_slc = slice(*s['mapping_range'])
                cal_ts = self.g_pool.timestamps[cal_slc]
                map_ts = self.g_pool.timestamps[map_slc]

                color = cygl_utils.RGBA(*s['color'][:3], 1.)
                if len(cal_ts):
                    cygl_utils.draw_rounded_rect((cal_ts[0], -4 * scale),
                                      (cal_ts[-1] - cal_ts[0], 8 * scale),
                                      corner_radius=0,
                                      color=color,
                                      sharpness=1.)
                if len(map_ts):
                    cygl_utils.draw_rounded_rect((map_ts[0], -scale),
                                      (map_ts[-1] - map_ts[0], 2 * scale),
                                      corner_radius=0,
                                      color=color,
                                      sharpness=1.)

                color = cygl_utils.RGBA(1., 1., 1., .5)
                if s['calibration_method'] == "natural_features":
                    cygl_utils.draw_x([(m['timestamp'], 0) for m in self.manual_ref_positions],
                           height=12 * scale, width=3 * pixel_to_time_fac / scale, thickness=scale, color=color)
                else:
                    cygl_utils.draw_bars([(m['timestamp'], 0) for m in self.circle_marker_positions],
                              height=12 * scale, thickness=scale, color=color)

                gl.glTranslatef(0, scale * self.timeline_line_height, 0)

    def draw_labels(self, width, height, scale):
        self.glfont.set_size(self.timeline_line_height * scale)
        for s in self.sections:
            self.glfont.draw_text(width, 0, s['label'])
            gl.glTranslatef(0, self.timeline_line_height * scale, 0)

    def toggle_marker_detection(self):
        if self.process_pipe:
            self.cancel_marker_detection()
        else:
            self.start_marker_detection()

    def start_marker_detection(self):
        self.circle_marker_positions = []
        source_path = self.g_pool.capture.source_path
        self.process_pipe = zmq_tools.Msg_Pair_Server(self.g_pool.zmq_ctx)
        self.notify_all({'subject': 'circle_detector_process.should_start',
                         'source_path': source_path, "pair_url": self.process_pipe.url})

        self.detection_progress = 0.
        self.menu_icon.indicator_stop = 0.
        self.toggle_detection_button.label = 'Cancel circle marker detection'

    def cancel_marker_detection(self):
        if self.process_pipe:
            self.process_pipe.send({'topic': 'terminate'})
            self.process_pipe.socket.close()
            self.process_pipe = None

            self.detection_progress = 0.
            self.menu_icon.indicator_stop = 0.
            self.toggle_detection_button.label = 'Start circle marker detection'

    def cleanup(self):
        for sec in self.sections:
            if 'bg_task' in sec:
                sec['bg_task'].cancel()
                del sec['bg_task']
            sec['gaze'].clear()
            sec['gaze_ts'].clear()
        self.save_offline_data()

    def save_offline_data(self):
        session_data = {}
        session_data['sections'] = []
        for s in self.sections:
            sec = s.copy()
            for key in ('bg_task', 'gaze', 'gaze_ts'):
                if key in sec:
                    del sec[key]
            session_data['sections'].append(sec)
        session_data['version'] = self.session_data_version
        session_data['manual_ref_positions'] = self.manual_ref_positions
        if self.detection_progress == 100.0:
            session_data['circle_marker_positions'] = self.circle_marker_positions
        else:
            session_data['circle_marker_positions'] = []
        cache_path = os.path.join(self.result_dir, 'offline_calibration_gaze')
        fm.save_object(session_data, cache_path)
        logger.info('Cached offline calibration data to {}'.format(cache_path))
