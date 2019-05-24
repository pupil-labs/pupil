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

import file_methods as fm
import player_methods as pm
from gaze_producer.gaze_producer_base import GazeProducerBase


class GazeFromRecording(GazeProducerBase):
    pretty_class_name = "Gaze From Recording"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.g_pool.gaze_positions = self._load_gaze_data()
        self.notify_all({"subject": "gaze_positions_changed"})

    def _load_gaze_data(self):
        gaze = fm.load_pldata_file(self.g_pool.rec_dir, "gaze")
        return pm.Bisector(gaze.data, gaze.timestamps)

    def init_ui(self):
        super().init_ui()
        self.menu.label = "Gaze Data  From Recording"
        self.menu.append(
            ui.Info_Text("Currently, gaze positions are loaded from the recording.")
        )
