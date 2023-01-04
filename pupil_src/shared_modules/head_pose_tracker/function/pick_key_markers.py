"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections

import numpy as np

KeyMarker = collections.namedtuple(
    "KeyMarker", ["frame_id", "marker_id", "verts", "bin"]
)

min_n_markers_per_frame = 2
max_n_same_markers_per_bin = 1
assert min_n_markers_per_frame >= 2
assert max_n_same_markers_per_bin >= 1

n_bins_x = 2
n_bins_y = 2
_bins_x = np.linspace(0, 1, n_bins_x + 1)[1:-1]
_bins_y = np.linspace(0, 1, n_bins_y + 1)[1:-1]


_n_frames_passed = 0


def run(markers_in_frame, all_key_markers, select_key_markers_interval=2):
    assert select_key_markers_interval >= 1

    if _decide_key_markers(
        markers_in_frame, all_key_markers, select_key_markers_interval
    ):
        return _get_key_markers(markers_in_frame)
    else:
        return []


def _decide_key_markers(markers_in_frame, all_key_markers, select_key_markers_interval):
    global _n_frames_passed

    _n_frames_passed += 1
    if _n_frames_passed >= select_key_markers_interval:
        _n_frames_passed = 0

        if len(markers_in_frame) >= min_n_markers_per_frame:
            if _check_bins_availability(markers_in_frame, all_key_markers):
                return True
    return False


def _check_bins_availability(markers_in_frame, all_key_markers):
    for marker in markers_in_frame:
        n_same_markers_in_bin = len(
            [
                key_marker
                for key_marker in all_key_markers
                if key_marker.marker_id == marker["id"]
                and key_marker.bin == _get_bin(marker)
            ]
        )
        # when there is one marker whose bin is available,
        # all markers in this frame are regarded as key_markers
        if n_same_markers_in_bin < max_n_same_markers_per_bin:
            return True

    return False


def _get_key_markers(markers_in_frame):
    return [
        KeyMarker(marker["timestamp"], marker["id"], marker["verts"], _get_bin(marker))
        for marker in markers_in_frame
    ]


def _get_bin(detection):
    centroid = detection["centroid"]
    bin_x = int(np.digitize(centroid[0], _bins_x))
    bin_y = int(np.digitize(centroid[1], _bins_y))
    return bin_x, bin_y
