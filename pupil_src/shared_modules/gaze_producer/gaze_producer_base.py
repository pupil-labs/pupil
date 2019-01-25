"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

from pyglui import ui

import player_methods as pm
from plugin import Producer_Plugin_Base


class GazeProducerBase(Producer_Plugin_Base):
    uniqueness = "by_base_class"
    order = 0.02
    icon_chr = chr(0xEC14)
    icon_font = "pupil_icons"

    def init_ui(self):
        self.add_menu()
        self.menu_icon.order = 0.3

        self.menu.append(self._create_plugin_selector())

    def _create_plugin_selector(self):
        gaze_producer_plugins = [
            p
            for p in self.g_pool.plugin_by_name.values()
            if issubclass(p, GazeProducerBase)
        ]
        gaze_producer_plugins.sort(key=lambda p: p.__name__)

        def open_plugin(p):
            self.notify_all({"subject": "start_plugin", "name": p.__name__})

        # TODO: better name for selector?
        return ui.Selector(
            "gaze_producer",
            setter=open_plugin,
            getter=lambda: self.__class__,
            selection=gaze_producer_plugins,
            labels=[p.pretty_class_name for p in gaze_producer_plugins],
            label="Gaze Producers",
        )

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        # TODO: comments or method extraction
        if "frame" in events:
            frame_idx = events["frame"].index
            window = pm.enclosing_window(self.g_pool.timestamps, frame_idx)
            events["gaze"] = self.g_pool.gaze_positions.by_ts_window(window)
