'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from plugin import System_Plugin_Base
from pyglui import ui
from calibration_routines import Calibration_Plugin, Gaze_Mapping_Plugin
from video_capture import Base_Manager, Base_Source


class Plugin_Manager(System_Plugin_Base):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.user_plugins = [p for p in g_pool.plugin_by_name.values()
                             if not issubclass(p, (System_Plugin_Base,
                                                   Base_Manager, Base_Source,
                                                   Calibration_Plugin,
                                                   Gaze_Mapping_Plugin))]
        # self.user_plugins.remove('Select to load')

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Plugin Manager'
        self.menu_icon.label_font = 'pupil_icons'
        self.menu_icon.label = chr(0xe8c0)
        self.menu_icon.order = .0

        def plugin_menu_entry(p):
            def setter(turn_on):
                if turn_on:
                    self.notify_all({'subject': 'start_plugin', 'name': p.__name__})
                else:
                    for p_inst in self.g_pool.plugins:
                        if p_inst.class_name == p.__name__:
                            p_inst.alive = False
                            break

            def getter():
                for p_inst in self.g_pool.plugins:
                    if p_inst.class_name == p.__name__:
                        return True
                return False

            return ui.Switch(p.__name__, label=p.__name__.replace('_', ' '),
                             setter=setter, getter=getter)

        for p in self.user_plugins:
            self.menu.append(plugin_menu_entry(p))

    def deinit_ui(self):
        self.remove_menu()
