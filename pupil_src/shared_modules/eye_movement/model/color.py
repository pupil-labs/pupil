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

    @abc.abstractmethod
    def to_bgr(self) -> "Color_BGR":
        ...

    @abc.abstractmethod
    def to_bgra(self) -> "Color_BGRA":
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

    def to_bgr(self) -> "Color_BGR":
        return Color_BGR(blue=self.blue, green=self.green, red=self.red)

    def to_bgra(self) -> "Color_BGRA":
        return self.to_rgba().to_bgra()


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

    def to_bgr(self) -> "Color_BGR":
        return self.to_rgb().to_bgr()

    def to_bgra(self) -> "Color_BGRA":
        return Color_BGRA(
            blue=self.blue, green=self.green, red=self.red, alpha=self.alpha
        )


class Color_BGR(Color):
    def __init__(self, blue: int, green: int, red: int):
        clip_int = lambda i: int(max(0, min(i, 255)))
        self._channels = (clip_int(blue), clip_int(green), clip_int(red))

    @property
    def blue(self) -> int:
        return self.channels[0]

    @property
    def green(self) -> int:
        return self.channels[1]

    @property
    def red(self) -> int:
        return self.channels[2]

    @property
    def channels(self) -> t.Tuple[int, int, int]:
        return self._channels

    def to_rgb(self) -> "Color_RGB":
        return Color_RGB(red=self.red, green=self.green, blue=self.blue)

    def to_rgba(self) -> "Color_RGBA":
        return self.to_rgb().to_rgba()

    def to_bgr(self) -> "Color_BGR":
        return Color_BGR(*self.channels)

    def to_bgra(self) -> "Color_BGRA":
        return self.to_rgb().to_bgra()


class Color_BGRA(Color):
    def __init__(self, blue: float, green: float, red: float, alpha: float):
        clip_float = lambda f: float(max(0.0, min(f, 1.0)))
        self._channels = (
            clip_float(blue),
            clip_float(green),
            clip_float(red),
            clip_float(alpha),
        )

    @property
    def blue(self) -> float:
        return self.channels[0]

    @property
    def green(self) -> float:
        return self.channels[1]

    @property
    def red(self) -> float:
        return self.channels[2]

    @property
    def alpha(self) -> float:
        return self.channels[3]

    @property
    def channels(self) -> t.Tuple[float, float, float, float]:
        return self._channels

    def to_rgb(self, bg_color: Color_RGB = ...) -> "Color_RGB":
        return self.to_rgba().to_rgb()

    def to_rgba(self) -> "Color_RGBA":
        return Color_RGBA(
            red=self.red, green=self.green, blue=self.blue, alpha=self.alpha
        )

    def to_bgr(self) -> "Color_BGR":
        return self.to_rgba().to_bgr()

    def to_bgra(self) -> "Color_BGRA":
        return Color_BGRA(*self.channels)


class Color_Palette(abc.ABC):
    @property
    @abc.abstractmethod
    def grey(self):
        ...

    @property
    @abc.abstractmethod
    def yellow(self):
        ...

    @property
    @abc.abstractmethod
    def green(self):
        ...

    @property
    @abc.abstractmethod
    def blue(self):
        ...

    @property
    @abc.abstractmethod
    def purple(self):
        ...


class Base_Color_Palette(Color_Palette):
    grey = Color_RGB(128, 128, 128)
    yellow = Color_RGB(255, 255, 0)
    green = Color_RGB(0, 128, 0)
    blue = Color_RGB(0, 128, 255)
    purple = Color_RGB(128, 0, 255)


class Defo_Color_Palette(Color_Palette):
    """
    https://flatuicolors.com/palette/defo
    """

    WET_ASPHALT = Color_RGB(52, 73, 94)
    grey = WET_ASPHALT

    SUN_FLOWER = Color_RGB(241, 196, 15)
    yellow = SUN_FLOWER

    NEPHRITIS = Color_RGB(39, 174, 96)
    green = NEPHRITIS

    BELIZE_HOLE = Color_RGB(41, 128, 185)
    blue = BELIZE_HOLE

    WISTERIA = Color_RGB(142, 68, 173)
    purple = WISTERIA
