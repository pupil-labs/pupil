"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
from types import SimpleNamespace

import cv2
import numpy as np

import apriltag
import file_methods as fm
import video_capture
from methods import normalize
from stdlib_utils import unique

logger = logging.getLogger(__name__)

apriltag_detector = apriltag.Detector()


def get_markers_data(detection, img_size, timestamp):
    return {
        "id": detection.tag_id,
        "verts": detection.corners[::-1].tolist(),
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
    timestamps,
    frame_index_range,
    frame_index_to_num_markers,
    shared_memory,
):
    batch_size = 30
    frame_start, frame_end = frame_index_range
    frame_indices = sorted(
        set(range(frame_start, frame_end + 1)) - set(frame_index_to_num_markers.keys())
    )
    if not frame_indices:
        return

    frame_count = frame_end - frame_start + 1
    shared_memory.progress = (frame_indices[0] - frame_start + 1) / frame_count
    yield None

    src = video_capture.File_Source(SimpleNamespace(), source_path, timing=None)

    queue = []
    for frame_index in frame_indices:
        shared_memory.progress = (frame_index - frame_start + 1) / frame_count
        timestamp = timestamps[frame_index]
        src.seek_to_frame(frame_index)
        frame = src.get_frame()

        detections = _detect(frame)
        if detections:
            serialized_dicts = [
                fm.Serialized_Dict(detection) for detection in detections
            ]
            queue.append((timestamp, serialized_dicts, frame_index, len(detections)))
        else:
            queue.append((timestamp, [fm.Serialized_Dict({})], frame_index, 0))

        if len(queue) >= batch_size:
            data = queue[:batch_size]
            del queue[:batch_size]
            yield data

    yield queue


def online_detection(frame):
    return _detect(frame)
