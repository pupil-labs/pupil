"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as T

from pupil_detectors import DetectorBase

from .detector_base_plugin import PupilDetectorPlugin


class DetectorDummyPlugin(PupilDetectorPlugin):
    uniqueness = "by_base_class"
    label = "disabled"
    identifier = "disabled"

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Dummy Detector"

    def __init__(self, g_pool, *args, **kwargs):
        super().__init__(g_pool=g_pool)

    def detect(self, frame):
        pass

    def namespaced_detector_properties(self) -> dict:
        return {}

    def on_resolution_change(self, old_size, new_size):
        pass
