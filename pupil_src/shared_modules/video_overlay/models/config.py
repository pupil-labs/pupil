"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from storage import StorageItem
from video_overlay.utils.constraints import (
    BooleanConstraint,
    ConstraintedPosition,
    ConstraintedValue,
    InclusiveConstraint,
)


class Configuration(StorageItem):
    version = 0

    def __init__(
        self,
        video_path=None,
        origin_x=0,
        origin_y=0,
        scale=0.6,
        alpha=0.8,
        hflip=False,
        vflip=False,
    ):
        self.video_path = video_path
        self.origin = ConstraintedPosition(origin_x, origin_y)
        self.scale = ConstraintedValue(scale, InclusiveConstraint(low=0.2, high=1.0))
        self.alpha = ConstraintedValue(alpha, InclusiveConstraint(low=0.1, high=1.0))
        self.hflip = ConstraintedValue(hflip, BooleanConstraint())
        self.vflip = ConstraintedValue(vflip, BooleanConstraint())

    @property
    def as_tuple(self):
        return (
            self.video_path,
            self.origin.x.value,
            self.origin.y.value,
            self.scale.value,
            self.alpha.value,
            self.hflip.value,
            self.vflip.value,
        )

    @staticmethod
    def from_tuple(tuple_):
        return Configuration(*tuple_)

    def as_dict(self):
        return {
            "video_path": self.video_path,
            "origin_x": self.origin.x.value,
            "origin_y": self.origin.y.value,
            "scale": self.scale.value,
            "alpha": self.alpha.value,
            "hflip": self.hflip.value,
            "vflip": self.vflip.value,
        }
