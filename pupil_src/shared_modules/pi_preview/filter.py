import logging
from pyglui import ui

from pi_preview._eur_filter import OneEuroFilter

logger = logging.getLogger(__name__)


class Filter:
    def __init__(self, enabled=True, freq=120, mincutoff=1.0, beta=100.0, dcutoff=1.0):
        self.enabled = enabled
        self._freq = freq
        self._mincutoff = mincutoff
        self._beta = beta
        self._dcutoff = dcutoff
        self.reinit_filter()

    def apply(self, gaze):
        if self.enabled:
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

    def add_ui_elements(self, menu):
        filter_menu = ui.Growing_Menu("Gaze Filter")
        filter_menu.collapsed = True
        filter_menu.append(ui.Switch("enabled", self))
        filter_menu.append(ui.Text_Input("freq", self))
        filter_menu.append(ui.Text_Input("mincutoff", self))
        filter_menu.append(ui.Text_Input("beta", self))
        filter_menu.append(ui.Text_Input("dcutoff", self))
        filter_menu.append(ui.Separator())
        menu.append(filter_menu)

    def get_init_dict(self):
        return {
            "enabled": self.enabled,
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
