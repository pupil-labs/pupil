"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import enum
import typing as T


class FrameFormat(enum.Enum):
    JPEG = "jpeg"
    YUV = "yuv"
    BGR = "bgr"
    GRAY = "gray"

    @property
    def label(self) -> str:
        if self is FrameFormat.JPEG:
            return "JPEG"
        if self is FrameFormat.YUV:
            return "YUV"
        if self is FrameFormat.BGR:
            return "BGR"
        if self is FrameFormat.GRAY:
            return "Gray Image"
        raise NotImplementedError(f"Unexpected frame format: {self}")

    @staticmethod
    def available_formats() -> T.Tuple["FrameFormat"]:
        return tuple(FrameFormat)
