'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file LICENSE, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from base_backend import Base_Backend
from ..source import UVC_Source
import uvc

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class UVC_Backend(Base_Backend):

    def __init__(self, g_pool, settings):
        super(UVC_Backend, self).__init__(g_pool, settings, should_load_settings=False)
        self.attempt_loading_settings(settings)

    def attempt_loading_settings(self,settings):
        if  not self.init_from_settings(settings):
            super(UVC_Backend, self).attempt_loading_settings(settings)

    @staticmethod
    def stream_error_class():
        return uvc.StreamError

    @staticmethod
    def source_type():
        return 'Local / UVC'

    def init_from_settings(self, settings):
        succesfull = super(UVC_Backend,self).init_from_settings(settings)
        if not succesfull and settings:
            name = settings.get('name')
            priority_list = [name] if name else settings.get('names',[])
            succesfull = self.set_active_source_with_priority(
                priority_list,
                settings = settings
            )
        return succesfull

    def list_sources(self):
        return uvc.device_list()

    def set_active_source_with_priority(self, names, settings=None):
        logger.debug('set_active_source_with_priority: %s, %s'%(names,settings))
        for name in names:
            if self.set_active_source_with_name(name,settings=settings):
                return True
        return False

    def set_active_source_with_name(self, name, settings=None):
        if self.active_source:
            settings = settings or self.active_source.settings
        succesfull = super(UVC_Backend,self).set_active_source_with_name(name,settings)
        if not succesfull:
            for dev in self.list_sources():
                if dev['name'] == name:
                    return self.set_active_source_with_id(dev['uid'], settings=settings)
        return succesfull

    def set_active_source_with_id(self, source_id, settings=None):
        succesfull = super(UVC_Backend,self).set_active_source_with_id(source_id, settings)
        if not succesfull:
            if not uvc.is_accessible(source_id):
                logger.error('Selected UVC source is already in use or not accessible.')
                return True

            if self.active_source:
                settings = settings or self.active_source.settings
                self.active_source.close()
            try:
                self.active_source = UVC_Source(self.g_pool, source_id, self.on_frame_size_change)
                self.active_source_id = source_id
                if settings:
                    self.active_source.settings = settings
            except Exception as e:
                logger.error('Initializing UVC source failed because of: %s'%str(e))
                return False
            if self.menu:
                self.active_source.init_gui(self.menu)
            return True
        return succesfull