"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import typing as T
from enum import IntEnum, auto
from time import monotonic, sleep

import gl_utils
import numpy as np
from plugin import Plugin
from pyglui import cygl, ui

logger = logging.getLogger(__name__)


class InitialisationError(Exception):
    pass


class StreamError(Exception):
    pass


class EndofVideoError(Exception):
    pass


class NoMoreVideoError(Exception):
    pass


class SourceMode(IntEnum):
    # NOTE: IntEnum is serializable with msgpack
    AUTO = auto()
    MANUAL = auto()


class Base_Source(Plugin):
    """Abstract source class

    All source objects are based on `Base_Source`.

    A source object is independent of its matching manager and should be
    initialisable without it.

    Initialization is required to succeed. In case of failure of the underlying capture
    the follow properties need to be readable:

    - name
    - frame_rate
    - frame_size

    The recent_events function is allowed to not add a frame to the `events` object.

    Attributes:
        g_pool (object): Global container, see `Plugin.g_pool`
    """

    uniqueness = "by_base_class"
    order = 0.0
    icon_chr = chr(0xE412)
    icon_font = "pupil_icons"

    @property
    def pretty_class_name(self):
        return "Video Source"

    def __init__(
        self,
        g_pool,
        *,
        source_mode: T.Optional[SourceMode] = None,
        **kwargs,
    ):
        super().__init__(g_pool)
        self.g_pool.capture = self
        self._recent_frame = None
        self._intrinsics = None

        # Three relevant cases for initializing source_mode:
        #   - Plugin started at runtime: use existing source mode in g_pool
        #   - Fresh start without settings: initialize to auto
        #   - Start with settings: will be passed as parameter, use those
        if not hasattr(self.g_pool, "source_mode"):
            self.g_pool.source_mode = source_mode or SourceMode.AUTO

        if not hasattr(self.g_pool, "source_managers"):
            # If for some reason no manager is loaded, we initialize this ourselves.
            self.g_pool.source_managers = []

    def add_menu(self):
        super().add_menu()
        self.menu_icon.order = 0.2

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Video Source"
        self.update_menu()

    def deinit_ui(self):
        self.remove_menu()

    def source_list(self):
        source_type = "Camera" if self.manual_mode else "Device"
        entries = [(None, f"Activate {source_type}")]

        for manager in self.g_pool.source_managers:
            if self.manual_mode:
                sources = manager.get_cameras()
            else:
                sources = manager.get_devices()

            for info in sources:
                entries.append((info, info.label))

        if len(entries) == 1:
            entries.append((None, f"No {source_type}s Found!"))

        return zip(*entries)

    def activate_source(self, source_info):
        if source_info is not None:
            source_info.activate()

    @property
    def manual_mode(self) -> bool:
        return self.g_pool.source_mode == SourceMode.MANUAL

    @manual_mode.setter
    def manual_mode(self, enable) -> None:
        new_mode = SourceMode.MANUAL if enable else SourceMode.AUTO
        if new_mode != self.g_pool.source_mode:
            logger.debug(f"Setting source mode: {new_mode.name}")
            self.notify_all({"subject": "backend.change_mode", "mode": new_mode})

    def on_notify(self, notification: T.Dict[str, T.Any]):
        subject = notification["subject"]

        if subject == "backend.change_mode":
            mode = SourceMode(notification["mode"])
            if mode != self.g_pool.source_mode:
                self.g_pool.source_mode = mode
                # redraw menu to close potentially open (and now incorrect) dropdown
                self.update_menu()

        elif subject == "eye_process.started":
            # Make sure to broadcast current source mode once to newly started eyes so
            # they are always in sync!
            if self.g_pool.app == "capture" and self.g_pool.process == "world":
                self.notify_all(
                    {"subject": "backend.change_mode", "mode": self.g_pool.source_mode}
                )

    def update_menu(self) -> None:
        """Update the UI for the source.

        Do not overwrite this in inherited classes. Use ui_elements() instead.
        """

        del self.menu[:]

        if self.manual_mode:
            self.menu.append(
                ui.Info_Text("Select a camera to use as input for this window.")
            )
        else:
            self.menu.append(
                ui.Info_Text(
                    "Select a Pupil Core headset from the list."
                    " Cameras will be automatically selected for world and eye windows."
                )
            )

        self.menu.append(
            ui.Selector(
                "selected_source",
                selection_getter=self.source_list,
                getter=lambda: None,
                setter=self.activate_source,
                label=" ",  # TODO: pyglui does not allow using no label at all
            )
        )

        if not self.manual_mode:
            self.menu.append(
                ui.Info_Text(
                    "Enable manual camera selection to choose a specific camera"
                    " as input for every window."
                )
            )

        self.menu.append(
            ui.Switch("manual_mode", self, label="Enable Manual Camera Selection")
        )

        source_settings = self.ui_elements()
        if source_settings:
            settings_menu = ui.Growing_Menu(f"Settings")
            settings_menu.extend(source_settings)
            self.menu.append(settings_menu)

    def ui_elements(self) -> T.List[ui.UI_element]:
        """Returns a list of ui elements with info and settings for the source."""
        return []

    def recent_events(self, events):
        """Returns None

        Adds events['frame']=Frame(args)
            Frame: Object containing image and time information of the current
            source frame.
        """
        raise NotImplementedError()

    def gl_display(self):
        if self._recent_frame is not None:
            frame = self._recent_frame
            if (
                frame.yuv_buffer is not None
                # TODO: Find a better solution than this:
                and getattr(self.g_pool, "display_mode", "") != "algorithm"
            ):
                self.g_pool.image_tex.update_from_yuv_buffer(
                    frame.yuv_buffer, frame.width, frame.height
                )
            else:
                self.g_pool.image_tex.update_from_ndarray(frame.bgr)
            gl_utils.glFlush()
        should_flip = getattr(self.g_pool, "flip", False)
        gl_utils.make_coord_system_norm_based(flip=should_flip)
        self.g_pool.image_tex.draw()
        if not self.online:
            cygl.utils.draw_gl_texture(np.zeros((1, 1, 3), dtype=np.uint8), alpha=0.4)
        gl_utils.make_coord_system_pixel_based(
            (self.frame_size[1], self.frame_size[0], 3), flip=should_flip
        )

    @property
    def name(self) -> str:
        raise NotImplementedError()

    def get_init_dict(self):
        return {"source_mode": self.g_pool.source_mode}

    @property
    def frame_size(self):
        """Summary
        Returns:
            tuple: 2-element tuple containing width, height
        """
        raise NotImplementedError()

    @property
    def frame_rate(self):
        """
        Returns:
            int/float: Frame rate
        """
        raise NotImplementedError()

    @property
    def jpeg_support(self):
        """
        Returns:
            bool: Source supports jpeg data
        """
        raise NotImplementedError()

    @property
    def online(self):
        """
        Returns:
            bool: Source is avaible and streaming images.
        """
        return True

    @property
    def intrinsics(self):
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, model):
        self._intrinsics = model


class Base_Manager(Plugin):
    """Abstract base class for source managers.

    Managers are plugins that enumerate and load accessible sources from different
    backends, e.g. locally USB-connected cameras.

    Supported sources can be either single cameras or whole devices. Identification and
    activation of sources works via SourceInfo (see below).
    """

    # backend managers are always loaded and need to be loaded before the sources
    order = -1

    def __init__(self, g_pool):
        super().__init__(g_pool)

        # register all instances in g_pool.source_managers list
        if not hasattr(g_pool, "source_managers"):
            g_pool.source_managers = []

        if self not in g_pool.source_managers:
            g_pool.source_managers.append(self)

    def get_devices(self) -> T.Sequence["SourceInfo"]:
        """Return source infos for all devices that the backend supports."""
        return []

    def get_cameras(self) -> T.Sequence["SourceInfo"]:
        """Return source infos for all cameras that the backend supports."""
        return []

    def activate(self, key: T.Any) -> None:
        """Activate a source (device or camera) by key from source info."""
        pass


class SourceInfo:
    """SourceInfo is a proxy for a source (camera or device) from a manager.

    Managers hand out source infos that can be activated from other places in the code.
    A manager needs to identify a source uniquely by a key.
    """

    def __init__(self, label: str, manager: Base_Manager, key: T.Any):
        self.label = label
        self.manager = manager
        self.key = key

    def activate(self) -> None:
        self.manager.activate(self.key)

    def __str__(self) -> str:
        return f"{self.label} - {self.manager.class_name}({self.key})"


class Playback_Source(Base_Source):
    def __init__(self, g_pool, timing="own", *args, **kwargs):
        """
        The `timing` argument defines the source's behavior during recent_event calls
            'own': Timing is based on recorded timestamps; uses own wait function;
                    used in Capture as an online source
            'external': Uses Seek_Control's current playback time to figure out
                    most appropriate frame; does not wait on its own
            None: Simply returns next frame as fast as possible; used for detectors
        """
        super().__init__(g_pool, *args, **kwargs)
        assert timing in (
            "external",
            "own",
            None,
        ), f"invalid timing argument: {timing}"
        self.timing = timing
        self.finished_sleep = 0.0
        self._recent_wait_ts = -1
        self.play = True

    def seek_to_frame(self, frame_idx):
        raise NotImplementedError()

    def get_frame_index(self):
        raise NotImplementedError()

    def get_frame(self):
        raise NotImplementedError()

    def get_frame_index_ts(self):
        idx = self.get_frame_index()
        return idx, self.timestamps[idx]

    def wait(self, timestamp):
        if timestamp == self._recent_wait_ts:
            sleep(1 / 60)  # 60 fps on pause
        elif self.finished_sleep:
            target_wait_time = timestamp - self._recent_wait_ts
            time_spent = monotonic() - self.finished_sleep
            target_wait_time -= time_spent
            if 1 > target_wait_time > 0:
                sleep(target_wait_time)
        self._recent_wait_ts = timestamp
        self.finished_sleep = monotonic()

    def get_init_dict(self):
        return dict(**super().get_init_dict(), timing=self.timing)
