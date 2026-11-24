"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from types import SimpleNamespace

import numpy as np
import pytest
from blink_detection import Offline_Blink_Detection


def test_consolidate_back_to_back_blinks():
    timestamps = np.arange(10.0, 19.0)
    pupil_data = np.array([{"timestamp": float(ts)} for ts in timestamps])
    detector = SimpleNamespace(
        g_pool=SimpleNamespace(timestamps=timestamps),
        timestamps=timestamps,
        response_classification=np.array([0, 1, 1, -1, -1, 1, 1, -1, 0]),
        filter_response=np.array([0, 0.6, 0.7, -0.8, -0.9, 0.9, 0.75, -0.8, 0]),
        _pupil_data=lambda: pupil_data,
        notify_all=lambda notification: None,
    )

    Offline_Blink_Detection.consolidate_classifications(detector)

    blinks = {blink["id"]: blink for blink in detector.g_pool.blinks}
    assert set(blinks) == {1, 2}
    assert (blinks[1]["start_timestamp"], blinks[1]["end_timestamp"]) == (11, 14)
    second = blinks[2]
    assert (second["start_timestamp"], second["end_timestamp"]) == (15, 17)
    assert second["duration"] == 2
    assert (second["start_frame_index"], second["end_frame_index"]) == (5, 7)
    assert list(second["base_data"]) == [{"timestamp": 15.0}, {"timestamp": 16.0}]
    assert second["confidence"] == pytest.approx(0.825)
