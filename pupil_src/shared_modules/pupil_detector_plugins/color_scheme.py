"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import functools
import typing as T

ComponentType = T.Union[int, float]


class Color(T.NamedTuple):
    """Three-component color class"""

    c0: ComponentType
    c1: ComponentType
    c2: ComponentType

    @classmethod
    def from_hex(cls, hex: str) -> "Color":
        # find: rgb tuple from hex string
        # example: 00ff00 -> (0, 255, 0)
        c0, c1, c2 = (int(hex[i : i + 2], 16) for i in range(0, 5, 2))
        return cls(c0, c1, c2)

    @property
    @functools.lru_cache(maxsize=1)
    def flip_c0_c2(self) -> "Color":
        """Converts RGB->BGR and vice versa."""
        return type(self)(self.c2, self.c1, self.c0)

    @property
    @functools.lru_cache(maxsize=1)
    def as_int(self) -> "Color":
        if isinstance(self.c0, int):
            return self
        return Color(int(self.c0 * 255), int(self.c1 * 255), int(self.c2 * 255))

    @property
    @functools.lru_cache(maxsize=1)
    def as_float(self) -> "Color":
        if isinstance(self.c0, float):
            return self
        return Color(self.c0 / 255, self.c1 / 255, self.c2 / 255)


PUPIL_ELLIPSE_2D = Color.from_hex("FAC800")
PUPIL_ELLIPSE_3D = Color.from_hex("F92500")
EYE_MODEL_OUTLINE_LONG_TERM_BOUNDS_IN = Color.from_hex("0777FA")
EYE_MODEL_OUTLINE_LONG_TERM_BOUNDS_OUT = Color.from_hex("80E0FF")
EYE_MODEL_OUTLINE_ULTRA_LONG_TERM_DEBUG = Color.from_hex("FF5C16")
EYE_MODEL_OUTLINE_SHORT_TERM_DEBUG = Color.from_hex("FFB876")
