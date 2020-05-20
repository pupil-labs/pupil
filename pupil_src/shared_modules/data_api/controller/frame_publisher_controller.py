"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import typing as T

from observable import Observable

from data_api.model import FrameFormat


logger = logging.getLogger(__name__)


class FramePublisherController(Observable):

    def on_frame_publisher_did_start(self, format: FrameFormat):
        pass

    def on_frame_publisher_did_stop(self):
        pass

    def __init__(self, format="jpeg", **kwargs):
        self.__format = None
        self.__did_warn_recently = False

        # TODO: Replace format str with FrameFormat
        self.format = format

    def get_init_dict(self):
        return {"format": self.format}

    def cleanup(self):
        self.on_frame_publisher_did_stop()

    @property
    def format(self):
        return self.__format

    @format.setter
    def format(self, value):
        self.__format = value #TODO: Validate type and value of new `value`
        self.on_frame_publisher_did_start(format=self.__format)

    def create_world_frame_dicts_from_frame(self, frame) -> T.List[dict]:
        if not frame:
            return []

        try:
            if self.format == "jpeg":
                data = frame.jpeg_buffer
            elif self.format == "yuv":
                data = frame.yuv_buffer
            elif self.format == "bgr":
                data = frame.bgr
            elif self.format == "gray":
                data = frame.gray
            assert data is not None

        except (AttributeError, AssertionError, NameError):
            if not self.__did_warn_recently:
                logger.warning(
                    '{}s are not compatible with format "{}"'.format(
                        type(frame), self.format
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
                "format": self.format,
                "__raw_data__": [blob],
            }
        ]
