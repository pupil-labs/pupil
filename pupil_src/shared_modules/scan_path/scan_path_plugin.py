"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

from observable import Observable
from plugin import Plugin
from pyglui import ui

from .scan_path_storage import ScanPathItem, ScanPathStorage
from .scan_path_background_task import ScanPathBackgroundTask


logger = logging.getLogger(__name__)


class ScanPathPlugin(Plugin, Observable):

    icon_chr = chr(0xE422)
    icon_font = "pupil_icons"
    order = 0.1

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Scan Path"

    def __init__(self, g_pool, timeframe=0.5):
        super().__init__(g_pool)

        self._timeframe = timeframe

        self._bg_task = ScanPathBackgroundTask(g_pool)
        self._bg_task.add_observer("on_task_started", self.on_scan_path_task_started)
        self._bg_task.add_observer("on_task_updated", self.on_scan_path_task_updated)
        self._bg_task.add_observer("on_task_failed", self.on_scan_path_task_failed)
        self._bg_task.add_observer("on_task_completed", self.on_scan_path_task_completed)

        self._recalculate()

    def get_init_dict(self):
        return {"timeframe": self._timeframe}

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Scan Path"

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        self._bg_task.process()

        frame = events.get("frame", None)

        if not frame:
            return

        if self._bg_task.is_running:
            # Don't publish results until the whole task is finished
            return

        events["scan_path_gaze"] = self._storage.get(frame.index)

    def on_notify(self, notification):
        pass

    def _recalculate(self):
        self._storage = ScanPathStorage(self.g_pool.rec_dir, self)
        self._bg_task.start(self._timeframe)

    def on_scan_path_task_started(self):
        self.menu_icon.indicator_stop = 0.0

    def on_scan_path_task_updated(self, progress, frame_index, gaze_datums, corrected_gaze_datums):
        self.menu_icon.indicator_stop = progress

        item = ScanPathItem(frame_index, corrected_gaze_datums)
        self._storage.add(item)

    def on_scan_path_task_failed(self, error):
        self.menu_icon.indicator_stop = 0.0
        raise error #FIXME

    def on_scan_path_task_completed(self):
        self.menu_icon.indicator_stop = 0.0
