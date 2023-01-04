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

from network_api.model import FrameFormat
from observable import Observable

logger = logging.getLogger(__name__)


class FramePublisherController(Observable):
    def on_format_changed(self):
        logger.debug(f"on_format_changed({self.__frame_format})")

    def __init__(self, format="jpeg", **kwargs):
        self.__frame_format = FrameFormat(format)
        self.__did_warn_recently = False

    def get_init_dict(self):
        return {"format": self.__frame_format.value}

    @property
    def frame_format(self):
        return self.__frame_format

    @frame_format.setter
    def frame_format(self, value):
        self.__frame_format = FrameFormat(value)
        self.on_format_changed()

    def create_world_frame_dicts_from_frame(self, frame) -> T.List[dict]:
        if not frame:
            return []

        try:
            if self.__frame_format == FrameFormat.JPEG:
                data = frame.jpeg_buffer
            elif self.__frame_format == FrameFormat.YUV:
                data = frame.yuv_buffer
            elif self.__frame_format == FrameFormat.BGR:
                data = frame.bgr
            elif self.__frame_format == FrameFormat.GRAY:
                data = frame.gray
            assert data is not None

        except (AttributeError, AssertionError, NameError):
            if not self.__did_warn_recently:
                logger.warning(
                    '{}s are not compatible with format "{}"'.format(
                        type(frame), self.__frame_format
                    )
                )
                self.__did_warn_recently = True
            return []
        else:
            self.__did_warn_recently = False

        # Create serializable object.
        # Not necessary if __raw_data__ key is used.
        # blob = memoryview(np.asarray(data).data)
        blob = data

        return [
            {
                "topic": "frame.world",
                "width": frame.width,
                "height": frame.height,
                "index": frame.index,
                "timestamp": frame.timestamp,
                "format": self.__frame_format.value,
                "__raw_data__": [blob],
            }
        ]
