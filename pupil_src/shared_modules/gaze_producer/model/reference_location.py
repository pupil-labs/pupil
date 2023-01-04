"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

from storage import StorageItem


@dataclass
class ReferenceLocation(StorageItem):
    screen_pos: Tuple[float, float]
    frame_index: int
    timestamp: float
    mm_pos: Optional[Tuple[float, float, float]] = None

    version: ClassVar[int] = 1

    @property
    def screen_x(self):
        return self.screen_pos[0]

    @property
    def screen_y(self):
        return self.screen_pos[1]

    @staticmethod
    def from_tuple(tuple_):
        return ReferenceLocation(*tuple_)

    @property
    def as_tuple(self):
        return self.screen_pos, self.frame_index, self.timestamp, self.mm_pos
