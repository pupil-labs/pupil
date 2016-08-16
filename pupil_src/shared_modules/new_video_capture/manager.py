'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file LICENSE, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from pyglui import ui
from backend import UVC_Backend
from backend import NDSI_Backend

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Manager(object):
    """docstring for Manager"""
    def __init__(self, g_pool, fallback_settings, previous_settings=None):
        super(Manager, self).__init__()
        self.g_pool = g_pool
        self.menu = None
        self.parent_menu = None
        self.source_type_selector = None

        self.active_backend = None
        self.active_backend_source_type = None
        self.source_types = {
            b.source_type() : b for b in [UVC_Backend, NDSI_Backend]
        }

        def do_nothing(*arg,**kwargs):
            pass
        self._on_frame_size_change = do_nothing
        if  not self.set_active_backend_from_settings(previous_settings) and \
            not self.set_active_backend_from_settings(fallback_settings):
            self.set_active_backend(UVC_Backend) # Fallback


    @property
    def on_frame_size_change(self):
        return self._on_frame_size_change
    @on_frame_size_change.setter
    def on_frame_size_change(self, fun):
        self._on_frame_size_change = fun
        if self.active_backend:
            self.active_backend.on_frame_size_change = fun

    @property
    def settings(self):
        settings = {}
        if self.active_backend:
            settings['active_backend'] = self.active_backend.settings
        return settings

    @property
    def frame_size(self):
        return self.settings['active_backend']['active_source']['frame_size']

    @property
    def frame_rate(self):
        return self.settings['active_backend']['active_source']['frame_rate']

    @property
    def settings(self):
        settings = {}
        if self.active_backend:
            settings['active_backend'] = self.active_backend.settings
        return settings

    def get_frame(self):
        return self.active_backend.get_frame()

    def set_active_backend_from_settings(self, settings):
        if not settings: return False
        actv_be = settings.get('active_backend')
        if actv_be and actv_be.get('source_type') in self.source_types:
            self.set_active_backend_by_source_type(actv_be['source_type'],actv_be.get('active_source'))
            return True
        return False

    def set_active_backend_by_source_type(self, src_type, settings=None):
        self.set_active_backend(self.source_types[src_type], settings=settings)

    def set_active_backend(self, backend, settings=None):
        """Sets active  backend

        Args:
            backend (class): Subclass of Base_Backend
        """
        if backend.source_type() not in self.source_types:
            raise ValueError('Attempt to initialize unknown backend "%s"'%backend.source_type())

        if self.active_backend_source_type == backend.source_type():
            return

        if not settings and self.active_backend:
            settings = self.active_backend.settings
        self.close_active_backend()
        if backend:
            # instanciates backend
            self.active_backend = backend(self.g_pool,settings)
            self.active_backend_source_type = backend.source_type()
            self.active_backend.on_frame_size_change = self.on_frame_size_change
            if self.menu:
                self.active_backend.init_gui(self.menu)

    def close_active_backend(self):
        if self.active_backend:
            self.active_backend.close()
            self.active_backend = None

    def init_gui(self, sidebar):
        self.parent_menu = sidebar
        self.menu = ui.Growing_Menu('Camera Settings')
        self.source_type_selector = ui.Selector(
            'active_backend_source_type',self,
            label            = 'Backend',
            setter           = self.set_active_backend_by_source_type,
            selection        = self.source_types.keys())
        self.menu.append(self.source_type_selector)
        if self.active_backend:
            self.active_backend.init_gui(self.menu)
        sidebar.insert(1,self.menu)

    def deinit_gui(self):
        if self.menu:
            if self.active_backend:
                self.active_backend.deinit_gui()
            if self.source_type_selector:
                self.menu.remove(self.source_type_selector)
                self.source_type_selector = None
            self.parent_menu.remove(self.menu)
            self.menu = None
            self.parent_menu = None

    def close(self):
        if self.menu:
            self.deinit_gui()
        self.close_active_backend()