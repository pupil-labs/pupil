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
import csv
from pyglui import ui
from plugin import Plugin, Analysis_Plugin_Base
from file_methods import load_object,save_object
from itertools import chain

import numpy as np
from OpenGL.GL import *
from glfw import glfwGetWindowSize,glfwGetCurrentContext
from pyglui.cygl.utils import draw_polyline,RGBA
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

# logging
import logging
logger = logging.getLogger(__name__)


class Annotation_Capture(Plugin):
    """Describe your plugin here
    """
    icon_chr = chr(0xe866)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool, annotations=(('My annotation', 'E'),)):
        super().__init__(g_pool)
        self.menu = None
        self.sub_menu = None
        self.buttons = []

        self.annotations = list(annotations)

        self.new_annotation_name = 'new annotation name'
        self.new_annotation_hotkey = 'e'

        self.current_frame = -1

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Annotations'
        self.menu.append(ui.Text_Input('new_annotation_name',self))
        self.menu.append(ui.Text_Input('new_annotation_hotkey',self))
        self.menu.append(ui.Button('add annotation type',self.add_annotation))
        self.sub_menu = ui.Growing_Menu('Events - click to remove')
        self.menu.append(self.sub_menu)
        self.update_buttons()

    def update_buttons(self):
        for b in self.buttons:
            self.g_pool.quickbar.remove(b)
            self.sub_menu.elements[:] = []
        self.buttons = []

        for e_name, hotkey in self.annotations:
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

    def add_annotation(self):
        self.annotations.append((self.new_annotation_name, self.new_annotation_hotkey))
        self.update_buttons()

    def remove_annotation(self, annotation):
        try:
            self.annotations.remove(annotation)
        except ValueError:
            print(annotation, self.annotations)
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
        return {'annotations': self.annotations}


class Annotation_Player(Annotation_Capture, Analysis_Plugin_Base):
    """Describe your plugin here
    View,edit and add Annotations.
    """
    def __init__(self,g_pool,annotations=None):
        if annotations:
            super().__init__(g_pool,annotations)
        else:
            super().__init__(g_pool)

        from player_methods import correlate_data

        #first we try to load annoations previously saved with pupil player
        try:
            annotations_list = load_object(os.path.join(self.g_pool.rec_dir, "annotations"))
        except IOError as e:
            #if that fails we assume this is the first time this recording is played and we load annotations from pupil_data
            try:
                notifications_list = load_object(os.path.join(self.g_pool.rec_dir, "pupil_data"))['notifications']
                annotations_list = [n for n in notifications_list if n['subject']=='annotation']
            except (KeyError,IOError) as e:
                annotations_list = []
                logger.debug('No annotations found in pupil_data file.')
            else:
                logger.debug('loaded {} annotations from pupil_data file'.format(len(annotations_list)))
        else:
            logger.debug('loaded {} annotations from annotations file'.format(len(annotations_list)))

        self.annotations_by_frame = correlate_data(annotations_list, self.g_pool.timestamps)
        self.annotations_list = annotations_list

    def init_ui(self):
        self.add_menu()
        # lets make a menu entry in the sidebar
        self.menu.label = 'View and edit annotations'
        self.menu.append(ui.Info_Text("Annotations recorded with capture are displayed when this plugin is loaded. New annotations can be added with the interface below."))
        self.menu.append(ui.Info_Text("If you want to revert annotations to the recorded state, stop player, delete the annotations file in the recording and reopen player."))

        self.menu.append(ui.Text_Input('new_annotation_name',self))
        self.menu.append(ui.Text_Input('new_annotation_hotkey',self))
        self.menu.append(ui.Button('add annotation type',self.add_annotation))
        self.sub_menu = ui.Growing_Menu('Events - click to remove')
        self.menu.append(self.sub_menu)
        self.update_buttons()

    def fire_annotation(self, annotation_label):
        t = self.last_frame_ts
        logger.info('"{}"@{}'.format(annotation_label, t))
        # you may add more field to this dictionary if you want.
        notification = {'subject': 'annotation', 'label': annotation_label,
                        'timestamp': t, 'duration': 0.0, 'added_in_player': True,
                        'index': self.g_pool.capture.get_frame_index()-1}
        self.annotations_list.append(notification)
        self.annotations_by_frame[notification['index']].append(notification)

    @classmethod
    def parse_csv_keys(self, annotations):
        csv_keys = ('index', 'timestamp', 'label', 'duration')
        system_keys = set(csv_keys)
        user_keys = set()
        for anno in annotations:
            # selects keys that are not included in system_keys and
            # adds them to user_keys if they were not included before
            user_keys |= set(anno.keys()) - system_keys

        # return tuple with system keys first and alphabetically sorted
        # user keys afterwards
        return csv_keys + tuple(sorted(user_keys))

    def export_annotations(self, export_range, export_dir):

        if not self.annotations:
            logger.warning('No annotations in this recording nothing to export')
            return

        start, end = export_range
        annotations_in_section = [a for a in self.annotations_list if start <= a['index'] < end]
        csv_keys = self.parse_csv_keys(annotations_in_section)

        with open(os.path.join(export_dir, 'annotations.csv'), 'w', encoding='utf-8', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_keys)
            for a in annotations_in_section:
                csv_writer.writerow((a.get(k, '') for k in csv_keys))
            logger.info("Created 'annotations.csv' file.")

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        self.last_frame_ts = frame.timestamp
        if frame.index != self.current_frame:
            self.current_frame = frame.index
            events = self.annotations_by_frame[frame.index]
            for e in events:
                logger.info(str(e))

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
            self.export_annotations(notification['range'],notification['export_dir'])

    def cleanup(self):
        """called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        save_object(self.annotations_list,os.path.join(self.g_pool.rec_dir, "annotations"))
