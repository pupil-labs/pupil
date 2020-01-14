"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import copy
import functools

import numpy as np
import cv2

import methods as m
import file_methods as fm


class ScanPathAlgorithm:
    def __init__(self, timeframe: float):
        assert timeframe

        # user settings
        self.timeframe = timeframe

        # algorithm working data
        self.reset()

    def reset(self):
        self._prev_frame_index = -1
        self._prev_gray_image = None
        self._prev_gaze_datums = []

    def update_from_frame(self, frame, gaze_datums):
        width, height = frame.width, frame.height
        return self.update_from_raw_data(
            frame_index=frame.index,
            gaze_datums=gaze_datums,
            image_size=(width, height),
            gray_image=frame.gray,
        )

    def update_from_raw_data(self, frame_index, gaze_datums, image_size, gray_image):
        gaze_datums = [{"norm_pos": (gd["norm_x"], gd["norm_y"]), "timestamp": gd["timestamp"]} for gd in gaze_datums]

        is_succeeding_frame = frame_index - self._prev_frame_index == 1
        assert is_succeeding_frame, "Must provide succeeding frames"

        normalize_point = functools.partial(m.normalize, size=image_size, flip_y=True)

        denormalize_point = functools.partial(m.denormalize, size=image_size, flip_y=True)

        updated_prev_gaze_datums = []

        # lets update past gaze using optical flow: this is like sticking the gaze points onto the pixels of the img.
        if self._prev_gaze_datums:
            prev_gaze_points = [denormalize_point(ng["norm_pos"]) for ng in self._prev_gaze_datums]
            prev_gaze_points = np.asarray(prev_gaze_points, dtype=np.float32)

            new_gaze_points, status, err = cv2.calcOpticalFlowPyrLK(
                self._prev_gray_image,
                gray_image,
                prev_gaze_points,
                None,
                **self._lk_params
            )

            results = zip(self._prev_gaze_datums, new_gaze_points, status, err)

            for gaze_datum, new_gaze_point, s, e in results:
                if s:
                    new_gaze_datum = fm._recursive_deep_copy(gaze_datum) #TODO: Maybe not that efficient
                    new_gaze_datum["norm_pos"] = normalize_point(new_gaze_point)
                    updated_prev_gaze_datums.append(new_gaze_datum)
                else:
                    # logger.debug("dropping gaze")
                    # Since we will replace self.past_gaze_positions later,
                    # not appedning tu updated_prev_gaze_datums is like deliting this data point.
                    pass
        else:
            pass #TODO: Handle case for first frame passed with no previous history

        # trim gaze that is too old
        if len(gaze_datums) > 0:
            now = gaze_datums[0]["timestamp"]
            cutoff = now - self.timeframe
            updated_prev_gaze_datums = [g for g in updated_prev_gaze_datums if g["timestamp"] > cutoff]

        # inject the scan path gaze points into recent_gaze_positions
        all_gaze_datums = updated_prev_gaze_datums + gaze_datums
        all_gaze_datums = list(map(dict, all_gaze_datums))
        all_gaze_datums.sort(key=lambda x: x["timestamp"])  # this may be redundant...
        all_gaze_datums = fm._recursive_deep_copy(all_gaze_datums)

        # update info for next frame.
        self._prev_gray_image = gray_image
        self._prev_frame_index = frame_index
        self._prev_gaze_datums = all_gaze_datums

        return all_gaze_datums

    # Private

    # vars for calcOpticalFlowPyrLK
    _lk_params = dict(
        winSize=(90, 90),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        minEigThreshold=0.005,
    )
