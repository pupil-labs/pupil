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
import typing as T
from pathlib import Path

import av
import numpy as np

import camera_models as cm

from ..recording import PupilRecording

logger = logging.getLogger(__name__)


def _try_patch_world_instrinsics_file(rec_dir: str, videos: T.Sequence[Path]) -> None:
    """Tries to create a reasonable world.intrinsics file from a set of videos."""
    if not videos:
        return

    # Make sure the default value always correlates to the frame size of BrokenStream
    frame_size = (1280, 720)
    # TODO: Due to the naming conventions for multipart-recordings, we can't
    # easily lookup 'any' video name in the pre_recorded_calibrations, since it
    # might be a multipart recording. Therefore we need to compute a hint here
    # for the lookup. This could be improved.
    camera_hint = ""
    for video in videos:
        try:
            container = av.open(str(video))
        except av.AVError:
            continue

        for camera in cm.pre_recorded_calibrations:
            if camera in video.name:
                camera_hint = camera
                break
        frame_size = (
            container.streams.video[0].format.width,
            container.streams.video[0].format.height,
        )
        break

    intrinsics = cm.load_intrinsics(rec_dir, camera_hint, frame_size)
    intrinsics.save(rec_dir, "world")


_ConversionCallback = T.Callable[[np.array], np.array]


def _rewrite_times(
    recording: PupilRecording,
    dtype: str,
    conversion: T.Optional[_ConversionCallback] = None,
) -> None:
    """Load raw times (assuming dtype), apply conversion and save as _timestamps.npy."""
    for path in recording.files().raw_time():
        timestamps = np.fromfile(str(path), dtype=dtype)

        if conversion is not None:
            timestamps = conversion(timestamps)

        new_name = f"{path.stem}_timestamps.npy"
        logger.info(f"Creating {new_name}")
        timestamp_loc = path.parent / new_name
        np.save(str(timestamp_loc), timestamps)
