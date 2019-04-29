"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import typing as t


class Color(abc.ABC):
    @abc.abstractmethod
    def to_rgb(self) -> "Color_RGB":
        ...

    @abc.abstractmethod
    def to_rgba(self) -> "Color_RGBA":
        ...


class Color_RGB(Color):
    def __init__(self, red: int, green: int, blue: int):
        clip_int = lambda i: int(max(0, min(i, 255)))
        self._channels = (clip_int(red), clip_int(green), clip_int(blue))

    @property
    def red(self) -> int:
        return self.channels[0]

    @property
    def green(self) -> int:
        return self.channels[1]

    @property
    def blue(self) -> int:
        return self.channels[2]

    @property
    def channels(self) -> t.Tuple[int, int, int]:
        return self._channels

    def to_rgb(self) -> "Color_RGB":
        return Color_RGB(*self.channels)

    def to_rgba(self) -> "Color_RGBA":
        r, g, b = self.channels
        return Color_RGBA(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0)


Color_RGB.BLACK = Color_RGB(0, 0, 0)
Color_RGB.WHITE = Color_RGB(255, 255, 255)


class Color_RGBA(Color):

    DEFAULT_RGB_BLEND_BG_COLOR = Color_RGB.BLACK

    def __init__(self, red: float, green: float, blue: float, alpha: float):
        clip_float = lambda f: float(max(0.0, min(f, 1.0)))
        self._channels = (
            clip_float(red),
            clip_float(green),
            clip_float(blue),
            clip_float(alpha),
        )

    @property
    def red(self) -> float:
        return self.channels[0]

    @property
    def green(self) -> float:
        return self.channels[1]

    @property
    def blue(self) -> float:
        return self.channels[2]

    @property
    def alpha(self) -> float:
        return self.channels[3]

    @property
    def channels(self) -> t.Tuple[float, float, float, float]:
        return self._channels

    def to_rgb(self, bg_color: Color_RGB = ...) -> "Color_RGB":
        bg_color = self.DEFAULT_RGB_BLEND_BG_COLOR if bg_color is ... else bg_color

        def blend_channel(src: float, bg: int, alpha: float) -> int:
            """
            https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
            """
            return int(((1.0 - alpha) * bg) + (alpha * src))

        red = blend_channel(self.red, bg_color.red, self.alpha)
        green = blend_channel(self.green, bg_color.green, self.alpha)
        blue = blend_channel(self.blue, bg_color.blue, self.alpha)

        return Color_RGB(red, green, blue)

    def to_rgba(self) -> "Color_RGBA":
        return Color_RGBA(*self._channels)


class Color_Palette(abc.ABC):
    pass


class Defo_Color_Palette(Color_Palette):
    """
    https://flatuicolors.com/palette/defo
    """

    WET_ASPHALT = Color_RGB(52, 73, 94)  # grey
    SUN_FLOWER = Color_RGB(241, 196, 15)  # yellow
    NEPHRITIS = Color_RGB(39, 174, 96)  # green
    BELIZE_HOLE = Color_RGB(41, 128, 185)  # blue
    WISTERIA = Color_RGB(142, 68, 173)  # purple
