"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import typing as T
from collections import namedtuple

from gaze_producer.model.calibration import Calibration

logger = logging.getLogger(__name__)


class CalibrationV1:
    version = 1

    Result = namedtuple("CalibrationResultV1", ["mapping_plugin_name", "mapper_args"])

    def __init__(
        self,
        unique_id,
        name,
        recording_uuid,
        mapping_method,
        frame_index_range,
        minimum_confidence,
        status="Not calculated yet",
        is_offline_calibration=True,
        result=None,
    ):
        self.unique_id = unique_id
        self.name = name
        self.recording_uuid = recording_uuid
        self.mapping_method = mapping_method
        self.frame_index_range = frame_index_range
        self.minimum_confidence = minimum_confidence
        self.status = status
        self.is_offline_calibration = is_offline_calibration
        if result is None:
            self.result = result
        else:
            self.result = CalibrationV1.Result(*result)

    @staticmethod
    def from_tuple(tuple_) -> "CalibrationV1":
        assert isinstance(tuple_, (list, tuple))  # sanity check
        return CalibrationV1(*tuple_)

    @property
    def updated_name(self) -> str:
        return f"{self.name} [UPDATED v{Calibration.version}]"

    def updated(self) -> Calibration:
        from gaze_mapping import Gazer2D, Gazer3D

        gazer_class, gazer_params = _gazer_class_and_params_from_gaze_mapper_result(
            self.result
        )

        if not self.is_offline_calibration:
            raise ValueError(
                f"Updating pre-recorded (read-only) calibrations is not supported. {self.name}"
            )

        is_offline_calibration = gazer_params is not None
        status = self.status if gazer_params is not None else "Not calculated yet"

        return Calibration(
            unique_id=self.unique_id,
            name=self.updated_name,
            recording_uuid=self.recording_uuid,
            gazer_class_name=gazer_class.__name__,
            frame_index_range=self.frame_index_range,
            minimum_confidence=self.minimum_confidence,
            status=status,
            is_offline_calibration=is_offline_calibration,
            calib_params=gazer_params,
        )


def _gazer_class_and_params_from_gaze_mapper_result(gaze_mapper_result):
    from gaze_mapping import Gazer2D, Gazer3D

    gazer_class = None
    gazer_params = None

    if gaze_mapper_result is None:
        return gazer_class, gazer_params

    gaze_mapper_name = gaze_mapper_result.mapping_plugin_name
    gaze_mapper_args = gaze_mapper_result.mapper_args

    deprecated_keys = {
        "cal_points_3d",
        "cal_ref_points_3d",
        "cal_gaze_points_3d",
        "cal_gaze_points0_3d",
        "cal_gaze_points1_3d",
    }
    gaze_mapper_args = {
        k: v for k, v in gaze_mapper_args.items() if k not in deprecated_keys
    }

    if gaze_mapper_name == "Binocular_Vector_Gaze_Mapper":
        gazer_class = Gazer3D

        if gaze_mapper_args is not None:
            assert set(gaze_mapper_args.keys()).issuperset(
                {
                    "eye_camera_to_world_matrix0",
                    "eye_camera_to_world_matrix1",
                }
            )  # sanity check
            gazer_params = {"binocular_model": gaze_mapper_args}

    elif gaze_mapper_name == "Vector_Gaze_Mapper":
        gazer_class = Gazer3D

        if gaze_mapper_args is not None:
            assert set(gaze_mapper_args.keys()).issuperset(
                {
                    "eye_camera_to_world_matrix",
                }
            )  # sanity check
            # Since there is no way to know which eye the mapper belongs to,
            # we use the arguments as the parameters for both eye models.
            gazer_params = {
                "left_model": gaze_mapper_args.copy(),
                "right_model": gaze_mapper_args.copy(),
            }

    elif gaze_mapper_name == "Binocular_Gaze_Mapper":
        gazer_class = Gazer2D
        # No way to extract the params from old calibration
        gazer_params = None

    elif gaze_mapper_name == "Monocular_Gaze_Mapper":
        gazer_class = Gazer2D
        # No way to extract the params from old calibration
        gazer_params = None

    elif gaze_mapper_name == "Dual_Monocular_Gaze_Mapper":
        gazer_class = Gazer2D
        # No way to extract the params from old calibration
        gazer_params = None

    else:
        logger.debug(
            f'Unable extract gazer class and params from gaze mapper "{gaze_mapper_name}"'
        )

    return gazer_class, gazer_params
