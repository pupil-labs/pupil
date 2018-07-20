'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import csv
# logging
import logging
import os
from itertools import chain

import numpy as np
from OpenGL.GL import *
from pyglui import ui
from pyglui.cygl.utils import RGBA, draw_polyline
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

import player_methods  as pm
import file_methods  as fm
from glfw import glfwGetCurrentContext, glfwGetWindowSize
from plugin import Analysis_Plugin_Base, Plugin

logger = logging.getLogger(__name__)


class Annotation_Capture(Plugin):
    """Describe your plugin here
    """
    icon_chr = chr(0xe866)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool, annotation_definitions=(('My annotation', 'E'),)):
        super().__init__(g_pool)
        self.menu = None
        self.sub_menu = None
        self.buttons = []

        self.annotation_definitions = list(annotation_definitions)

        self.new_annotation_name = 'new annotation name'
        self.new_annotation_hotkey = 'e'

        self.current_frame = -1

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Annotations'
        self.menu.append(ui.Text_Input('new_annotation_name',self))
        self.menu.append(ui.Text_Input('new_annotation_hotkey',self))
        self.menu.append(ui.Button('add annotation type',self.append_annotation))
        self.sub_menu = ui.Growing_Menu('Events - click to remove')
        self.menu.append(self.sub_menu)
        self.update_buttons()

    def update_buttons(self):
        for b in self.buttons:
            self.g_pool.quickbar.remove(b)
            self.sub_menu.elements[:] = []
        self.buttons = []

        for e_name, hotkey in self.annotation_definitions:
            def make_fire(e_name, hotkey):
                return lambda _ : self.fire_annotation(e_name)

            def make_remove(e_name, hotkey):
                return lambda: self.remove_annotation((e_name, hotkey))

            thumb = ui.Thumb(e_name, setter=make_fire(e_name, hotkey),
                             getter=lambda: False, label=hotkey, hotkey=hotkey)
            self.buttons.append(thumb)
            self.g_pool.quickbar.append(thumb)
            self.sub_menu.append(ui.Button(e_name+" <"+hotkey+">", make_remove(e_name, hotkey)))

    def deinit_ui(self):
        self.menu.remove(self.sub_menu)
        self.sub_menu = None
        self.remove_menu()
        if self.buttons:
            for b in self.buttons:
                self.g_pool.quickbar.remove(b)
            self.buttons = []

    def append_annotation(self):
        self.annotation_definitions.append((self.new_annotation_name,
                                            self.new_annotation_hotkey))
        self.update_buttons()

    def remove_annotation(self, annotation):
        try:
            self.annotation_definitions.remove(annotation)
        except ValueError:
            print(annotation, self.annotation_definitions)
        self.update_buttons()

    def fire_annotation(self, annotation_label):
        t = self.g_pool.get_timestamp()
        logger.info('"{}"@{}'.format(annotation_label, t))
        # you may add more field to this dictionary if you want.
        notification = {'subject': 'annotation', 'label': annotation_label,
                        'timestamp': t, 'duration': 0.0, 'record': True}
        self.notify_all(notification)

    def on_notify(self, notification):
        if notification['subject'] == 'annotation':
            logger.info('Received {} annotation'.format(notification['label']))

    def get_init_dict(self):
        return {'annotation_definitions': self.annotation_definitions}


class Annotation_Player(Annotation_Capture, Analysis_Plugin_Base):
    """Describe your plugin here
    View,edit and add Annotations.
    """
    def __init__(self, g_pool, *args, **kwargs):
        super().__init__(g_pool, *args, **kwargs)
        self.cache_dir = os.path.join(g_pool.rec_dir, 'offline_data')
        cache_file = os.path.join(self.cache_dir, 'annotations.pldata')
        if os.path.exists(cache_file):
            self.load_cached_annotations()
        else:
            self.extract_annotations_from_recorded_notifications()

    def load_cached_annotations(self):
        annotations = fm.load_pldata_file(self.cache_dir, 'annotations')
        self.annotations = pm.Mutable_Bisector(annotations.data,
                                               annotations.timestamps)
        logger.info('Loaded {} annotations from cache.'.format(len(self.annotations)))

    def extract_annotations_from_recorded_notifications(self):
        notifications = fm.load_pldata_file(self.g_pool.rec_dir, 'notify')
        annotation_ts = []
        annotation_data = []
        for idx, topic in enumerate(notifications.topics):
            if topic == 'notify.annotation':
                annotation_ts.append(notifications.timestamps[idx])
                annotation_data.append(notifications.data[idx])
        self.annotations = pm.Mutable_Bisector(annotation_data, annotation_ts)
        logger.info('Extracted {} annotations from recording.'.format(len(self.annotations)))

    def init_ui(self):
        self.add_menu()
        # lets make a menu entry in the sidebar
        self.menu.label = 'View and edit annotations'
        self.menu.append(ui.Info_Text("Annotations recorded with capture are displayed when this plugin is loaded. New annotations can be added with the interface below."))
        self.menu.append(ui.Info_Text("If you want to revert annotations to the recorded state, stop player, delete the annotations file in the recording and reopen player."))

        self.menu.append(ui.Text_Input('new_annotation_name',self))
        self.menu.append(ui.Text_Input('new_annotation_hotkey',self))
        self.menu.append(ui.Button('add annotation type',self.append_annotation))
        self.sub_menu = ui.Growing_Menu('Events - click to remove')
        self.menu.append(self.sub_menu)
        self.update_buttons()

    def fire_annotation(self, annotation_label):
        t = self.last_frame_ts
        logger.info('"{}"@{}'.format(annotation_label, t))
        # you may add more field to this dictionary if you want.
        annotation_new = {'subject': 'annotation',
                          'topic': 'notify.annotation',
                          'label': annotation_label,
                          'timestamp': t, 'duration': 0.0,
                          'added_in_player': True}
        self.annotations.insert(annotation_new['timestamp'],
                                fm.Serialized_Dict(python_dict=annotation_new))

    @classmethod
    def parse_csv_keys(self, annotations):
        csv_keys = ('index', 'timestamp', 'label', 'duration')
        system_keys = set(csv_keys)
        user_keys = set()
        for anno in annotations:
            # selects keys that are not included in system_keys and
            # adds them to user_keys if they were not included before
            user_keys |= set(anno.keys()) - system_keys

        blacklisted_keys = set(('subject', 'topic', 'record'))
        user_keys -= blacklisted_keys

        # return tuple with system keys first and alphabetically sorted
        # user keys afterwards
        return csv_keys + tuple(sorted(user_keys))

    def export_annotations(self, export_range, export_dir):

        if not self.annotations:
            logger.warning('No annotations in this recording nothing to export')
            return

        export_window = pm.exact_window(self.g_pool.timestamps, export_range)
        annotation_section = self.annotations.init_dict_for_window(export_window)
        annotation_idc = pm.find_closest(self.g_pool.timestamps, annotation_section['data_ts'])
        csv_keys = self.parse_csv_keys(annotation_section['data'])

        with open(os.path.join(export_dir, 'annotations.csv'), 'w', encoding='utf-8', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_keys)
            for annotation, idx in zip(annotation_section['data'], annotation_idc):
                csv_row = [idx]
                csv_row.extend((annotation.get(k, '') for k in csv_keys[1:]))
                csv_writer.writerow(csv_row)
            logger.info("Created 'annotations.csv' file.")

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        self.last_frame_ts = frame.timestamp
        if frame.index != self.current_frame:
            self.current_frame = frame.index
            frame_window = pm.enclosing_window(self.g_pool.timestamps, frame.index)
            events = self.annotations.by_ts_window(frame_window)
            for e in events:
                logger.info('Annotation "{}"@{}'.format(e['label'], e['timestamp']))

    def deinit_ui(self):
        self.menu.remove(self.sub_menu)
        self.sub_menu = None
        self.remove_menu()

        if self.buttons:
            for b in self.buttons:
                self.g_pool.quickbar.remove(b)
            self.buttons = []

    def on_notify(self,notification):
        if notification['subject'] == "should_export":
            self.export_annotations(notification['range'],
                                    notification['export_dir'])

    def cleanup(self):
        """called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        with fm.PLData_Writer(self.cache_dir, 'annotations') as writer:
            for ts, annotation in zip(self.annotations.timestamps, self.annotations):
                writer.append_serialized(ts, 'notify.annotation', annotation.serialized)
