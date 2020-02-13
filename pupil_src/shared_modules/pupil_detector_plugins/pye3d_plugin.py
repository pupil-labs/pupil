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

from pye3d.eyemodel import EyeModel_V2 as EyeModel


from .detector_base_plugin import PupilDetectorPlugin

logger = logging.getLogger(__name__)


class Pye3DPlugin(PupilDetectorPlugin):
    uniqueness = "by_class"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC19)

    label = "Pye3D"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.detector = EyeModel()

    def detect(self, frame, pupil_data):
        for datum in pupil_data:
            if datum.get("method", "") == "2d c++":
                datum_2d = datum
                break
        else:
            return None

        datum_2d["raw_edges"] = []
        result = self.detector.update_and_detect(datum_2d)

        eye_id = self.g_pool.eye_id
        result["timestamp"] = frame.timestamp
        result["topic"] = f"pupil.{eye_id}"
        result["id"] = eye_id
        result["method"] = "3d c++"

        return result

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Pye3D Detector"
