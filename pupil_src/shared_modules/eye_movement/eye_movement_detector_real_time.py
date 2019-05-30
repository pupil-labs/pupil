"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from .eye_movement_detector_base import Eye_Movement_Detector_Base
import eye_movement.utils as utils
import eye_movement.model as model
import eye_movement.worker as worker
import eye_movement.ui as ui
from pyglui import ui as gl_ui
from pyglui.pyfontstash import fontstash


MIN_DATA_CONFIDENCE_DEFAULT: float = 0.6


class Eye_Movement_Detector_Real_Time(Eye_Movement_Detector_Base):
    """
    Eye movement classification detector based on segmented linear regression.

    Event identification is based on segmentation that simultaneously denoises the signal and determines event
    boundaries. The full gaze position time-series is segmented into an approximately optimal piecewise linear
    function in O(n) time. Gaze feature parameters for classification into fixations, saccades, smooth pursuits and post-saccadic oscillations
    are derived from human labeling in a data-driven manner.

    More details about this approach can be found here:
    https://www.nature.com/articles/s41598-017-17983-x

    The open source implementation can be found here:
    https://gitlab.com/nslr/nslr-hmm
    """

    MENU_LABEL_TEXT = "Eye Movement Detector"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self._buffered_detector = worker.Real_Time_Buffered_Detector()
        self._recent_segments = []
        self.glfont = None

    @property
    def min_data_confidence(self):
        try:
            return self.g_pool.min_data_confidence
        except AttributeError:
            return MIN_DATA_CONFIDENCE_DEFAULT

    def recent_events(self, events):

        gaze_data = events["gaze"]
        capture = model.Immutable_Capture(self.g_pool.capture)

        min_data_confidence = self.min_data_confidence
        gaze_data = [
            datum for datum in gaze_data if datum["confidence"] > min_data_confidence
        ]

        self._buffered_detector.extend_gaze_data(gaze_data=gaze_data, capture=capture)

        if "frame" not in events:
            return

        frame_timestamp = events["frame"].timestamp
        self._recent_segments = self._buffered_detector.segments_at_timestamp(
            frame_timestamp
        )

        public_segments = [
            segment.to_public_dict() for segment in self._recent_segments
        ]
        events[utils.EYE_MOVEMENT_EVENT_KEY] = public_segments

    def gl_display(self):
        segment_renderer = ui.Segment_Overlay_GL_Context_Renderer(
            canvas_size=self.g_pool.capture.frame_size, gl_font=self.glfont
        )
        for segment in self._recent_segments:
            segment_renderer.draw(segment)

    def init_ui(self):
        self.add_menu()
        self.menu.label = type(self).MENU_LABEL_TEXT

        for help_block in self.__doc__.split("\n\n"):
            help_str = help_block.replace("\n", " ").replace("  ", "").strip()
            self.menu.append(gl_ui.Info_Text(help_str))

        color_legend_text = ui.color_from_segment.__doc__
        color_legend_text = color_legend_text.strip()
        color_legend_text = color_legend_text.replace("  ", "")
        color_legend_text = color_legend_text.replace("\n-", "\n    -")
        self.menu.append(gl_ui.Info_Text(color_legend_text))

        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", gl_ui.get_opensans_font_path())

    def deinit_ui(self):
        self.remove_menu()
        self.glfont = None
