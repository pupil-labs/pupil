'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from . import Base_Manager
from ..source import Fake_Source

class Fake_Manager(Base_Manager):
    """Simple manager to explicitly activate a fake source"""

    gui_name = 'Test image'

    def __init__(self, g_pool):
        super(Fake_Manager, self).__init__(g_pool)

    def init_gui(self):
        from pyglui import ui
        text = ui.Info_Text('Convenience manager to select a fake source explicitly.')
        def activate():
            settings = self.g_pool.capture.settings
            self.activate_source(Fake_Source, settings)
        activation_button = ui.Button('Activate Test Image', activate)
        self.g_pool.capture_selector_menu.extend([text, activation_button])