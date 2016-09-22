'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
from plugin import Plugin
from ..source import InitialisationError, Fake_Source

import logging
logger = logging.getLogger(__name__)

class Base_Backend(Plugin):
    """docstring for Base_Backend"""

    uniqueness = 'by_base_class'

    def __init__(self, g_pool):
        super(Base_Backend, self).__init__(g_pool)
        g_pool.capture_backend = self

    def get_init_dict(self):
        return {}

    def activate_source(self, source_class, settings={}):
        prev_source_class = None
        if self.g_pool.capture:
            prev_source_class = self.g_pool.capture.__class__
            prev_settings = self.g_pool.capture.settings
            self.g_pool.capture.deinit_gui()
            self.g_pool.capture.cleanup()
            self.g_pool.capture = None
        try:
            self.g_pool.capture = source_class(self.g_pool,**settings)
        except InitialisationError:
            # try to recover to previous source
            if prev_source_class and prev_source_class != Fake_Source:
                self.activate_source(prev_source_class,prev_settings)
                logger.info('Reactivated `%s`'%prev_settings['name'])
            # no recovery possible, load fake source
            else: self.g_pool.capture = Fake_Source(self.g_pool,**settings)
        self.g_pool.capture.init_gui()

    def init_gui(self):
        self.g_pool.capture_selector_menu.extend([])

    def deinit_gui(self):
        del self.g_pool.capture_selector_menu[1:]

    def cleanup(self):
        self.deinit_gui()

    def on_notify(self,n):
        """Provides UI for the capture selection

        Handles notifications:
            ``capture_backend.source_found``

        Emmits notifications:
            ``capture_backend.source_found``
            ``capture_backend.source_lost``
        """
        from .. import source_classes
        source_class_by_name = {sc.class_name():sc for sc in source_classes}

        if (n['subject'].startswith('capture_backend.source_found')
            and self.g_pool.capture.class_name() == Fake_Source.class_name()):
            preferred = self.g_pool.capture.preferred_source
            found_source_class_name = n['source_class_name']
            preferred_source_class_name = preferred['source_class_name']
            if found_source_class_name == preferred_source_class_name:
                source_class = source_class_by_name[found_source_class_name]
                self.activate_source(source_class, preferred)


from fake_backend import Fake_Backend
from uvc_backend  import UVC_Backend
#from ndsi_backend import NDSI_Backend