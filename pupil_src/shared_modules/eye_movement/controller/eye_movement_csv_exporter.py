"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing as t
from csv_utils import CSV_Exporter, CSV_EXPORT_SCHEMA_TYPE
from eye_movement.utils import logger
from eye_movement.model.segment import Classified_Segment


class Eye_Movement_CSV_Exporter(CSV_Exporter[Classified_Segment]):

    EXPORT_FILE_NAME = "eye_movement.csv"

    @classmethod
    def csv_export_schema(cls) -> CSV_EXPORT_SCHEMA_TYPE:
        return [
            ("id", lambda seg: seg.id),
            ("method", lambda seg: seg.method.value),
            ("segment_class", lambda seg: seg.segment_class.value),
            ("start_frame_index", lambda seg: seg.start_frame_index),
            ("end_frame_index", lambda seg: seg.end_frame_index),
            ("start_timestamp", lambda seg: seg.start_frame_timestamp),
            ("end_timestamp", lambda seg: seg.end_frame_timestamp),
            ("duration", lambda seg: seg.duration),
            ("confidence", lambda seg: seg.confidence),
            ("norm_pos_x", lambda seg: seg.norm_pos[0]),
            ("norm_pos_y", lambda seg: seg.norm_pos[1]),
            (
                "gaze_point_3d_x",
                lambda seg: seg.gaze_point_3d[0] if seg.gaze_point_3d else "",
            ),
            (
                "gaze_point_3d_y",
                lambda seg: seg.gaze_point_3d[1] if seg.gaze_point_3d else "",
            ),
            (
                "gaze_point_3d_z",
                lambda seg: seg.gaze_point_3d[2] if seg.gaze_point_3d else "",
            ),
        ]

    def csv_export(
        self,
        segments: t.Iterable[Classified_Segment],
        export_dir: str,
        export_name: str = ...,
    ):

        export_name = self.EXPORT_FILE_NAME if export_name is ... else export_name
        export_path = super().csv_export(
            raw_values=segments, export_dir=export_dir, export_name=export_name
        )

        logger.info("Created file: '{}'.".format(export_path))
        return export_path
