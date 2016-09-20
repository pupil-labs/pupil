'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file LICENSE, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from pyglui import ui
from ..source import Fake_Source

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Base_Backend(object):
    """docstring for Base_Backend"""
    def __init__(self, g_pool, settings, should_load_settings=True):
        super(Base_Backend, self).__init__()
        self.g_pool = g_pool
        self.active_source = None
        self.active_source_id = 0xDeadBeef # None is for fake source
        self.menu = None
        self.parent_menu = None
        self.source_selector = None

        self._on_frame_size_change = None

        if should_load_settings:
            self.attempt_loading_settings(settings)

    def attempt_loading_settings(self,settings):
        if not self.init_from_settings(settings):
            self.init_with_fake_source()

    @property
    def on_frame_size_change(self):
        return self._on_frame_size_change
    @on_frame_size_change.setter
    def on_frame_size_change(self, fun):
        self._on_frame_size_change = fun
        if self.active_source:
            self.active_source.on_frame_size_change = fun

    @property
    def name(self):
        return type(self).__name__

    @staticmethod
    def stream_error_class():
        return None

    @classmethod
    def source_type(self):
        return self.__name__

    @property
    def settings(self):
        settings = {
            'source_type': self.source_type()
        }
        if self.active_source:
            settings['active_source'] = self.active_source.settings
        return settings


    def init_from_settings(self, settings):
        logger.debug('Try init from settings: %s'%settings)
        try:
            actv_src_settings = settings['active_source']
            return self.set_active_source_with_name(actv_src_settings['name'], settings=actv_src_settings)
        except:
            logger.debug('Settings do not contain previous source.')
            return False

    def init_with_fake_source(self):
        logger.debug('Init fake source.')
        return self.set_active_source_with_id(None)

    def list_sources(self):
        return []

    def get_frame(self):
        return self.active_source.get_frame()

    def set_active_source_with_name(self, name, settings=None):
        if not name:
            return self.set_active_source_with_id(None)
        else:
            return False

    def set_active_source_with_id(self, source_id, settings=None):
        if self.active_source_id == source_id:
            return True
        if not source_id:
            if self.active_source:
                settings = settings or self.active_source.settings
                self.active_source.close()
            self.active_source = Fake_Source(self.g_pool)
            self.active_source_id = None
            if settings:
                self.active_source.settings = settings
            if self.menu:
                self.active_source.init_gui(self.menu)
            self.active_source.on_frame_size_change = self.on_frame_size_change
            logger.info('Activated fake source with settings: %s'%settings)
            return True
        else:
            return False

    def init_gui(self,parent_menu,insert_index=-1):
        # init menu
        self.parent_menu = parent_menu
        self.menu = ui.Growing_Menu(self.name)

        def selection_getter():
            sources = self.list_sources()
            source_names = ['Fake Capture']+[s['name'] for s in sources]
            source_ids = [None]+[s['uid'] for s in sources]
            return source_ids, source_names

        self.source_selector = ui.Selector(
            'active_source_id',self,
            label            = 'Source',
            setter           = self.set_active_source_with_id,
            selection_getter = selection_getter
        )
        self.menu.append(self.source_selector)

        if self.active_source:
            self.active_source.init_gui(self.menu)

        if insert_index < 0:
            parent_menu.append(self.menu)
        else:
            parent_menu.insert(self.menu, insert_index)

    def deinit_gui(self):
        if self.menu:
            if self.active_source:
                self.active_source.deinit_gui()
            if self.source_selector:
                self.menu.remove(self.source_selector)
                self.source_selector = None
            self.parent_menu.remove(self.menu)
            self.menu = None
            self.parent_menu = None

    def close(self):
        if self.menu:
            self.deinit_gui()
        if self.active_source:
            self.active_source.close()
            self.active_source = None
            self.active_source_id = None