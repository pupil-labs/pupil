"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

from plugin import Plugin
from pyglui import ui

from .model import FrameFormat
from .controller import FramePublisherController
from .controller import PupilRemoteController
from .ui import FramePublisherMenu
from .ui import PupilRemoteMenu


logger = logging.getLogger(__name__)


class DataApiPlugin(Plugin):
    menu_label = "Data API"
    icon_chr = chr(0xE307)
    icon_font = "pupil_icons"
    order = 0.01  # excecute first

    @property
    def pretty_class_name(self):
        return self.menu_label

    def __init__(self, g_pool, **kwargs):
        super().__init__(g_pool)
        self.__frame_publisher = FramePublisherController(**kwargs)
        self.__frame_publisher.add_observer("on_frame_publisher_did_stop", self.on_frame_publisher_did_stop)


        self.__pupil_remote = PupilRemoteController(g_pool, **kwargs)

        # UI
        self.__frame_publisher_ui_menu = FramePublisherMenu(self.__frame_publisher)
        self.__pupil_remote_ui_menu = PupilRemoteMenu()

    def get_init_dict(self):
        return {
            **self.__frame_publisher.get_init_dict(),
        }

    def cleanup(self):
        self.__frame_publisher.cleanup()
        self.__frame_publisher = None

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.menu_label
        self.frame_publisher_ui_menu.append_to_menu(self.menu)

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        frame = events.get("frame")
        if frame:
            world_frame_dicts = self.__frame_publisher.create_world_frame_dicts_from_frame(frame)
            if world_frame_dicts:
                events["frame.world"] = world_frame_dicts

    def on_notify(self, notification):
        """Publishes frame data in several formats

        Reacts to notifications:
            ``eye_process.started``: Re-emits ``frame_publishing.started``
            ``frame_publishing.set_format``: Sets image format specified in ``format`` field

        Emits notifications:
            ``frame_publishing.started``: Frame publishing started
            ``frame_publishing.stopped``: Frame publishing stopped
            ``recording.should_start``
            ``recording.should_stop``
            ``calibration.should_start``
            ``calibration.should_stop``
            Any other notification received though the reqrepl port.
        """
        if notification["subject"].startswith("eye_process.started"):
            # trigger notification
            self.__frame_publisher.format = self.__frame_publisher.format
        elif notification["subject"] == "frame_publishing.set_format":
            # update format and trigger notification
            self.__frame_publisher.format = notification["format"]

    def on_frame_publisher_did_start(self, format: FrameFormat):
        self.notify_all({"subject": "frame_publishing.started", "format": format.value})

    def on_frame_publisher_did_stop(self):
        self.notify_all({"subject": "frame_publishing.stopped"})
