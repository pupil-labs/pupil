import abc
import logging
from pyglui import ui

from pi_preview._eur_filter import OneEuroFilter

logger = logging.getLogger(__name__)


class Filter(abc.ABC):
    label = None

    def __init__(self, *, enabled=True):
        self.enabled = enabled

    def get_init_dict(self):
        return {"enabled": self.enabled}

    def apply(self, gaze):
        if self.enabled:
            self._apply(gaze)

    @abc.abstractmethod
    def _apply(self, gaze):
        raise NotImplementedError

    def add_ui_elements(self, menu):
        submenu = ui.Growing_Menu(self.label)
        submenu.collapsed = True
        submenu.append(ui.Switch("enabled", self, label="Enable"))
        self._add_ui_elements(submenu)
        submenu.append(ui.Button("Reset settings", self.reset))
        menu.append(submenu)

    def _add_ui_elements(self, menu):
        pass

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError


class OffsetFilter(Filter):
    label = "Offset Filter"

    def __init__(self, *, offset=(0.0, 0.0), **kwargs):
        super().__init__(**kwargs)
        self.offset = offset

    def _apply(self, gaze):
        x, y = gaze["norm_pos"]
        x += self.offset[0]
        y += self.offset[1]
        gaze["norm_pos"] = x, y

    def get_init_dict(self):
        return {**super().get_init_dict(), "offset": self.offset}

    def _add_ui_elements(self, menu):
        pass

    def reset(self):
        self.offset = (0.0, 0.0)


class TemporalFilter(Filter):
    label = "Temporal Filter"

    def __init__(self, *, freq=120, mincutoff=1.0, beta=100.0, dcutoff=1.0, **kwargs):
        super().__init__(**kwargs)
        self._freq = freq
        self._mincutoff = mincutoff
        self._beta = beta
        self._dcutoff = dcutoff
        self.reinit_filter()

    def reset(self):
        self._freq = 120
        self._mincutoff = 1.0
        self._beta = 100.0
        self._dcutoff = 1.0
        self.reinit_filter()

    def _apply(self, gaze):
        x, y = gaze["norm_pos"]
        ts = gaze["timestamp"]
        x = self.x_filter(x, ts)
        y = self.y_filter(y, ts)
        gaze["norm_pos"] = x, y

    def reinit_filter(self):
        config = {
            "freq": self._freq,
            "mincutoff": self._mincutoff,
            "beta": self._beta,
            "dcutoff": self._dcutoff,
        }
        self.x_filter = OneEuroFilter(**config)
        self.y_filter = OneEuroFilter(**config)

    def _add_ui_elements(self, menu):
        menu.append(ui.Text_Input("freq", self))
        menu.append(ui.Text_Input("mincutoff", self))
        menu.append(ui.Text_Input("beta", self))
        menu.append(ui.Text_Input("dcutoff", self))

    def get_init_dict(self):
        return {
            **super().get_init_dict(),
            "freq": self.freq,
            "mincutoff": self.mincutoff,
            "beta": self.beta,
            "dcutoff": self.dcutoff,
        }

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, val):
        if val <= 0:
            logger.error("freq should be >0")
        self._freq = val
        self.reinit_filter()

    @property
    def mincutoff(self):
        return self._mincutoff

    @mincutoff.setter
    def mincutoff(self, val):
        if val <= 0:
            logger.error("mincutoff should be >0")
        self._mincutoff = val
        self.reinit_filter()

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, val):
        self._beta = val
        self.reinit_filter()

    @property
    def dcutoff(self):
        return self._dcutoff

    @dcutoff.setter
    def dcutoff(self, val):
        if val <= 0:
            logger.error("dcutoff should be >0")
        self._dcutoff = val
        self.reinit_filter()
