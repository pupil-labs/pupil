"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc

import data_changed
import player_methods as pm
from observable import Observable
from plugin import System_Plugin_Base
from pyglui import ui


class GazeProducerBase(Observable, System_Plugin_Base):
    uniqueness = "by_base_class"
    order = 0.02
    icon_chr = chr(0xEC14)
    icon_font = "pupil_icons"

    @classmethod
    @abc.abstractmethod
    def plugin_menu_label(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def gaze_data_source_selection_label(cls) -> str:
        return cls.plugin_menu_label()

    @classmethod
    def gaze_data_source_selection_order(cls) -> float:
        return float("inf")

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self._gaze_changed_announcer = data_changed.Announcer(
            "gaze_positions", g_pool.rec_dir, plugin=self
        )

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.plugin_menu_label()
        self.menu_icon.order = 0.3
        self.menu_icon.tooltip = "Gaze Data"

        self.menu.append(self._create_plugin_selector())

    def _create_plugin_selector(self):
        gaze_producer_plugins = [
            p
            for p in self.g_pool.plugin_by_name.values()
            if issubclass(p, GazeProducerBase)
        ]
        # Skip gaze producers that are not available within g_pool context
        gaze_producer_plugins = [
            p
            for p in gaze_producer_plugins
            if p.is_available_within_context(self.g_pool)
        ]
        gaze_producer_plugins.sort(key=lambda p: p.gaze_data_source_selection_label())
        gaze_producer_plugins.sort(key=lambda p: p.gaze_data_source_selection_order())
        gaze_producer_labels = [
            p.gaze_data_source_selection_label() for p in gaze_producer_plugins
        ]

        def open_plugin(p):
            self.notify_all({"subject": "start_plugin", "name": p.__name__})

        # TODO: better name for selector?
        return ui.Selector(
            "gaze_producer",
            setter=open_plugin,
            getter=lambda: self.__class__,
            selection=gaze_producer_plugins,
            labels=gaze_producer_labels,
            label="Data Source",
        )

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        # TODO: comments or method extraction
        if "frame" in events:
            frame_idx = events["frame"].index
            window = pm.enclosing_window(self.g_pool.timestamps, frame_idx)
            events["gaze"] = self.g_pool.gaze_positions.by_ts_window(window)
