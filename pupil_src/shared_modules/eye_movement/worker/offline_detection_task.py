"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import typing

import nslr_hmm
import numpy as np

import eye_movement.model as model
import eye_movement.utils as utils
import file_methods as fm
from tasklib.background.task import TypedBackgroundGeneratorFunction

# TODO: Replace `str` with `Eye_Movement_Detection_Step`
Offline_Detection_Task_Yield = typing.Tuple[
    str, typing.Optional[utils.MsgPack_Serialized_Segment]
]

EYE_MOVEMENT_DETECTION_STEP_PREPARING_LOCALIZED_STRING = "Preparing gaze data..."
EYE_MOVEMENT_DETECTION_STEP_PROCESSING_LOCALIZED_STRING = "Processing gaze data..."
EYE_MOVEMENT_DETECTION_STEP_CLASSIFYING_LOCALIZED_STRING = "Classifying gaze data..."
EYE_MOVEMENT_DETECTION_STEP_DETECTING_LOCALIZED_STRING = (
    "Detecting segmentation events..."
)
EYE_MOVEMENT_DETECTION_STEP_COMPLETE_LOCALIZED_STRING = "Segmentation complete"


Offline_Detection_Task_Generator = typing.Iterator[Offline_Detection_Task_Yield]
Offline_Detection_Task_Args = typing.Tuple[model.Immutable_Capture, utils.Gaze_Data]
Offline_Detection_Task_Function = typing.Callable[
    [model.Immutable_Capture, utils.Gaze_Data], Offline_Detection_Task_Generator
]


@typing.no_type_check
def eye_movement_detection_generator(
    capture: model.Immutable_Capture,
    gaze_data: utils.Gaze_Data,
    factory_start_id: int = None,
) -> Offline_Detection_Task_Generator:
    def serialized_dict(datum):
        if type(datum) is dict:
            return fm.Serialized_Dict(python_dict=datum)
        elif type(datum) is bytes:
            return fm.Serialized_Dict(msgpack_bytes=datum)
        else:
            raise ValueError("Unsupported gaze datum type: {}.".format(type(datum)))

    yield EYE_MOVEMENT_DETECTION_STEP_PREPARING_LOCALIZED_STRING, ()
    gaze_data = [serialized_dict(datum) for datum in gaze_data]

    if not gaze_data:
        utils.logger.warning("No data available to find fixations")
        yield EYE_MOVEMENT_DETECTION_STEP_COMPLETE_LOCALIZED_STRING, ()
        return

    use_pupil = utils.can_use_3d_gaze_mapping(gaze_data)

    segment_factory = model.Classified_Segment_Factory(start_id=factory_start_id)

    gaze_time = np.array([gp["timestamp"] for gp in gaze_data])

    yield EYE_MOVEMENT_DETECTION_STEP_PROCESSING_LOCALIZED_STRING, ()
    eye_positions = utils.gaze_data_to_nslr_data(
        capture, gaze_data, gaze_time, use_pupil=use_pupil
    )

    yield EYE_MOVEMENT_DETECTION_STEP_CLASSIFYING_LOCALIZED_STRING, ()
    gaze_classification, segmentation, segment_classification = nslr_hmm.classify_gaze(
        gaze_time, eye_positions
    )

    # `gaze_classification` holds the classification for each gaze datum.

    yield EYE_MOVEMENT_DETECTION_STEP_DETECTING_LOCALIZED_STRING, ()
    for i, nslr_segment in enumerate(segmentation.segments):

        nslr_segment_class = segment_classification[i]

        segment = segment_factory.create_segment(
            gaze_data=gaze_data,
            gaze_time=gaze_time,
            use_pupil=use_pupil,
            nslr_segment=nslr_segment,
            nslr_segment_class=nslr_segment_class,
            world_timestamps=capture.timestamps,
        )

        if not segment:
            continue

        serialized = segment.to_msgpack()

        yield EYE_MOVEMENT_DETECTION_STEP_DETECTING_LOCALIZED_STRING, serialized

    yield EYE_MOVEMENT_DETECTION_STEP_COMPLETE_LOCALIZED_STRING, ()


class Offline_Detection_Task(
    TypedBackgroundGeneratorFunction[Offline_Detection_Task_Yield, None, None]
):
    def __init__(
        self,
        args: Offline_Detection_Task_Args,
        name: str = "Offline_Eye_Movement_Detection_Task",
        generator_function: Offline_Detection_Task_Function = eye_movement_detection_generator,
    ):
        super().__init__(name=name, generator_function=generator_function, args=args)
