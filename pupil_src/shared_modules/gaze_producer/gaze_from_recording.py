"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from pyglui import ui

import file_methods as fm
import player_methods as pm
from gaze_producer.gaze_producer_base import GazeProducerBase


class GazeFromRecording(GazeProducerBase):
    @classmethod
    def plugin_menu_label(cls) -> str:
        return "Gaze Data From Recording"

    @classmethod
    def gaze_data_source_selection_order(cls) -> float:
        return 1.0

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.g_pool.gaze_positions = self._load_gaze_data()
        self._gaze_changed_announcer.announce_existing()

    def _load_gaze_data(self):
        gaze = fm.load_pldata_file(self.g_pool.rec_dir, "gaze")
        return pm.Bisector(gaze.data, gaze.timestamps)

    def init_ui(self):
        super().init_ui()
        self.menu.append(ui.Info_Text("Using gaze data recorded by Pupil Capture."))
