"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

from calibration_choreography import CalibrationChoreographyPlugin
from gaze_mapping.gazer_base import GazerBase
from plugin import System_Plugin_Base
from pyglui import ui
from video_capture import Base_Manager, Base_Source

logger = logging.getLogger(__name__)


class Plugin_Manager(System_Plugin_Base):
    icon_chr = chr(0xE8C0)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        non_user_plugins = (
            System_Plugin_Base,
            Base_Manager,
            Base_Source,
            CalibrationChoreographyPlugin,
            GazerBase,
        )
        all_available_plugins = sorted(
            g_pool.plugin_by_name.values(), key=lambda p: p.__name__.lower()
        )

        available_and_supported_user_plugins = []

        for plugin in all_available_plugins:
            if issubclass(plugin, non_user_plugins):
                continue
            if not plugin.is_available_within_context(g_pool):
                logger.debug(
                    f"Plugin {plugin.__name__} not available; skip adding it to plugin list."
                )
                continue
            available_and_supported_user_plugins.append(plugin)

        self.user_plugins = available_and_supported_user_plugins

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Plugin Manager"
        self.menu_icon.order = 0.0

        def plugin_toggle_entry(p):
            def setter(turn_on):
                if turn_on:
                    self.notify_all({"subject": "start_plugin", "name": p.__name__})
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

            return ui.Switch(
                p.__name__,
                label=p.parse_pretty_class_name(),
                setter=setter,
                getter=getter,
            )

        def plugin_add_entry(p):
            def action():
                self.notify_all({"subject": "start_plugin", "name": p.__name__})

            return ui.Button("Add", action, p.__name__.replace("_", " "))

        if self.g_pool.app == "player":
            for p in self.user_plugins:
                if p.uniqueness != "not_unique":
                    self.menu.append(plugin_toggle_entry(p))
            self.menu.append(ui.Separator())
            for p in self.user_plugins:
                if p.uniqueness == "not_unique":
                    self.menu.append(plugin_add_entry(p))
        else:
            for p in self.user_plugins:
                self.menu.append(plugin_toggle_entry(p))

    def deinit_ui(self):
        self.remove_menu()
