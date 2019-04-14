from video_overlay.utils.constraints import (
    ConstraintedPosition,
    ConstraintedValue,
    BooleanConstraint,
    InclusiveConstraint,
)


class Controller:
    __slots__ = ("origin", "scale", "alpha", "hflip", "vflip")

    @classmethod
    def from_updated_defaults(cls, config_subset):
        defaults = cls.default_dict()
        defaults.update(config_subset)
        return cls(**defaults)

    @staticmethod
    def default_dict():
        return {
            "origin_x": 0,
            "origin_y": 0,
            "scale": 1.0,
            "alpha": 1.0,
            "hflip": False,
            "vflip": False,
        }

    def __init__(self, origin_x, origin_y, scale, alpha, hflip, vflip):
        self.origin = ConstraintedPosition(origin_x, origin_y)
        self.scale = ConstraintedValue(scale, InclusiveConstraint(low=0.2, high=1.0))
        self.alpha = ConstraintedValue(alpha, InclusiveConstraint(low=0.1, high=1.0))
        self.hflip = ConstraintedValue(hflip, BooleanConstraint())
        self.vflip = ConstraintedValue(vflip, BooleanConstraint())

    def get_init_dict(self):
        return {
            "origin_x": self.origin.x.value,
            "origin_y": self.origin.y.value,
            "scale": self.scale.value,
            "alpha": self.alpha.value,
            "hflip": self.hflip.value,
            "vflip": self.vflip.value,
        }
