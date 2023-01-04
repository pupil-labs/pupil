"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import file_methods as fm
import player_methods as pm
from gaze_producer.gaze_producer_base import GazeProducerBase
from pupil_recording import PupilRecording, RecordingInfo
from pyglui import ui


class GazeFromRecording(GazeProducerBase):
    @classmethod
    def is_available_within_context(cls, g_pool) -> bool:
        if g_pool.app == "player":
            recording = PupilRecording(rec_dir=g_pool.rec_dir)
            meta_info = recording.meta_info
            if (
                meta_info.recording_software_name
                == RecordingInfo.RECORDING_SOFTWARE_NAME_PUPIL_MOBILE
            ):
                # Disable gaze from recording in Player if Pupil Mobile recording
                return False
        return super().is_available_within_context(g_pool)

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
