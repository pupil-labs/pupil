'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from . import Base_Manager
from ..source import UVC_Source
import uvc, time
from sets import ImmutableSet

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class UVC_Manager(Base_Manager):

    gui_name = 'Local USB'

    def __init__(self, g_pool):
        super(UVC_Manager, self).__init__(g_pool)
        self.last_check_ts = 0.
        self.last_check_result = {}
        self.check_intervall = .5

    def init_gui(self):
        from pyglui import ui
        ui_elements = []
        ui_elements.append(ui.Info_Text('Local UVC sources'))

        def dev_selection_list():
            default = (None, 'Select to activate')
            devices = uvc.device_list()
            dev_pairs = [default] + [(d['uid'], d['name']) for d in devices]
            return zip(*dev_pairs)

        def activate(source_uid):
            if not source_uid:
                return
            settings = {
                'source_class_name': UVC_Source.class_name(),
                'frame_size': self.g_pool.capture.frame_size,
                'frame_rate': self.g_pool.capture.frame_rate,
                'uid': source_uid
            }
            self.activate_source(UVC_Source, settings)

        ui_elements.append(ui.Selector(
            'selected_source',
            selection_getter=dev_selection_list,
            getter=lambda: None,
            setter=activate,
            label='Activate source'
        ))
        self.g_pool.capture_selector_menu.extend(ui_elements)

    def update(self, frame, events):
        now = time.time()
        if now - self.last_check_ts > self.check_intervall:
            self.last_check_ts = now
            devices = uvc.device_list()
            device_names_by_uid = {d['uid']:d['name'] for d in devices}

            old_result = ImmutableSet(self.last_check_result.keys())
            new_result = ImmutableSet(device_names_by_uid.keys())

            for lost_key in old_result - new_result:
                self.notify_all({
                    'subject': 'capture_manager.source_lost',
                    'source_class_name': UVC_Source.class_name(),
                    'name': self.last_check_result[lost_key],
                    'uid': lost_key
                })
                del self.last_check_result[lost_key]

            for found_key in new_result - old_result:
                device_name = device_names_by_uid[found_key]
                self.notify_all({
                    'subject': 'capture_manager.source_found',
                    'source_class_name': UVC_Source.class_name(),
                    'name': device_name,
                    'uid': found_key
                })
                self.last_check_result[found_key] = device_name