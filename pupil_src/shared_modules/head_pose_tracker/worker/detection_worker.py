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
from types import SimpleNamespace

import cv2
import file_methods as fm
import numpy as np
import pupil_apriltags
import video_capture
from methods import normalize
from stdlib_utils import unique

logger = logging.getLogger(__name__)

apriltag_detector = pupil_apriltags.Detector(nthreads=2)


def get_markers_data(detection, img_size, timestamp):
    return {
        "id": detection.tag_id,
        # verts: Corners need to be listed counter-clockwise
        "verts": detection.corners.tolist(),
        "centroid": normalize(detection.center, img_size, flip_y=True),
        "timestamp": timestamp,
    }


def calc_perimeter(corners):
    corners = np.asarray(corners, dtype=np.float32)
    return cv2.arcLength(corners, closed=True)


def dedupliciate_markers(marker_old, marker_new):
    """Deduplicate markers by returning marker with bigger perimeter

    This heuristic is useful to remove "echos", i.e. markers that are being detected
    within the scene video preview instead of the scene itself.
    """
    perimeter_old = calc_perimeter(marker_old.corners)
    perimeter_new = calc_perimeter(marker_new.corners)
    return marker_old if perimeter_old > perimeter_new else marker_new


def _detect(frame):
    image = frame.gray
    apriltag_detections = apriltag_detector.detect(image)
    apriltag_detections = unique(
        apriltag_detections,
        key=lambda marker: marker.tag_id,
        select=dedupliciate_markers,
    )

    img_size = image.shape[::-1]
    return [
        get_markers_data(detection, img_size, frame.timestamp)
        for detection in apriltag_detections
    ]


def offline_detection(
    source_path,
    all_timestamps,
    frame_index_range,
    calculated_frame_indices,
    shared_memory,
):
    batch_size = 30
    frame_start, frame_end = frame_index_range
    frame_indices = sorted(
        set(range(frame_start, frame_end + 1)) - calculated_frame_indices
    )
    if not frame_indices:
        return

    frame_count = frame_end - frame_start + 1
    shared_memory.progress = (frame_indices[0] - frame_start + 1) / frame_count
    yield None

    src = video_capture.File_Source(
        SimpleNamespace(), source_path, fill_gaps=False, timing=None
    )
    timestamps_no_gaps = src.timestamps
    uncalculated_timestamps = all_timestamps[frame_indices]
    seek_poses = np.searchsorted(timestamps_no_gaps, uncalculated_timestamps)

    queue = []
    for frame_index, timestamp, target_frame_idx in zip(
        frame_indices, uncalculated_timestamps, seek_poses
    ):
        detections = []
        if timestamp in timestamps_no_gaps:
            if target_frame_idx != src.target_frame_idx:
                src.seek_to_frame(target_frame_idx)  # only seek frame if necessary
            frame = src.get_frame()
            detections = _detect(frame)

        serialized_dicts = [fm.Serialized_Dict(d) for d in detections]
        queue.append((timestamp, serialized_dicts, frame_index))

        if len(queue) >= batch_size:
            shared_memory.progress = (frame_index - frame_start + 1) / frame_count

            data = queue[:batch_size]
            del queue[:batch_size]
            yield data

    yield queue


def online_detection(frame):
    return _detect(frame)
