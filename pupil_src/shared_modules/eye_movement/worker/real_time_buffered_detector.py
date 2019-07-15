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
import bisect
import eye_movement.utils as utils
import eye_movement.model as model
import stdlib_utils
import numpy as np
import nslr_hmm


class Real_Time_Buffered_Detector:
    def __init__(self, max_segment_count: int = 1, max_sample_count: int = 1000):
        self._capture = None
        self._gaze_data_buffer = stdlib_utils.sliceable_deque(
            [], maxlen=max_sample_count
        )
        self._segment_buffer = stdlib_utils.sliceable_deque(
            [], maxlen=max_segment_count
        )
        self._segment_factory = model.Classified_Segment_Factory()
        self._is_gaze_buffer_classified: bool = True

    def extend_gaze_data(
        self, gaze_data: utils.Gaze_Data, capture: model.Immutable_Capture
    ):
        if not gaze_data:
            return
        self._capture = capture
        self._gaze_data_buffer.extend(gaze_data)
        #TODO: Remove manual sorting when timestamps are guaranteed to be monotonic
        #See: https://github.com/pupil-labs/pupil/issues/1493
        self._gaze_data_buffer = sorted(self._gaze_data_buffer, key=lambda gp: gp['timestamp'])
        self._is_gaze_buffer_classified = False

    def segments_at_timestamp(
        self, target_timestamp: float
    ) -> t.Iterable[model.Classified_Segment]:
        self._update_classification()
        return [
            segment
            for segment in self._segment_buffer
            if segment.time_range.contains(target_timestamp)
        ]

    def segments_in_time_range(
        self, target_range: model.Time_Range
    ) -> t.Iterable[model.Classified_Segment]:
        self._update_classification()
        return [
            segment
            for segment in self._segment_buffer
            if segment.time_range.intersection(target_range)
        ]

    @staticmethod
    def _segment_generator(
        capture: model.Immutable_Capture,
        gaze_data: utils.Gaze_Data,
        factory_start_id: int = None,
    ):
        # TODO: Merge this version with the one in offline_detection_task

        if len(gaze_data) < 2:
            utils.logger.warning("Not enough data available to find fixations")
            return

        use_pupil = utils.can_use_3d_gaze_mapping(gaze_data)

        segment_factory = model.Classified_Segment_Factory(start_id=factory_start_id)

        gaze_time = np.array([gp["timestamp"] for gp in gaze_data])

        try:
            eye_positions = utils.gaze_data_to_nslr_data(
                capture, gaze_data, gaze_time, use_pupil=use_pupil
            )
        except utils.NSLRValidationError as e:
            utils.logger.error(f"{e}")
            return

        gaze_classification, segmentation, segment_classification = nslr_hmm.classify_gaze(
            gaze_time, eye_positions
        )

        # by-gaze clasification, modifies events["gaze"] by reference
        for gaze, classification in zip(gaze_data, gaze_classification):
            gaze[utils.EYE_MOVEMENT_GAZE_KEY] = model.Segment_Class.from_nslr_class(
                classification
            ).value

        # by-segment classification
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

            yield segment

    def _update_classification(self):
        if self._is_gaze_buffer_classified:
            return

        factory_start_id = (
            self._segment_buffer[0].id if len(self._segment_buffer) > 0 else None
        )

        segment_generator = type(self)._segment_generator(
            capture=self._capture,
            gaze_data=self._gaze_data_buffer,
            factory_start_id=factory_start_id,
        )
        new_segments = list(segment_generator)

        # Update segment buffer by removing old segments and pushing new ones
        self._segment_buffer.clear()
        self._segment_buffer.extend(new_segments)

        # Update gaze data buffer by removing any datapoints that precede the first classified segment
        if len(self._segment_buffer) > 0:
            gaze_time_buffer = [gp["timestamp"] for gp in self._gaze_data_buffer]
            start_timestamp = self._segment_buffer[0].start_frame_timestamp
            i = bisect.bisect_left(gaze_time_buffer, start_timestamp)
            self._gaze_data_buffer = self._gaze_data_buffer[i:]

        # Mark current gaze data buffer as classified
        self._is_gaze_buffer_classified = True
