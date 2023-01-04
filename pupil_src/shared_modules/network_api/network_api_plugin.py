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

from plugin import Plugin

from .controller import FramePublisherController, PupilRemoteController
from .ui import FramePublisherMenu, PupilRemoteMenu

logger = logging.getLogger(__name__)


class NetworkApiPlugin(Plugin):
    menu_label = "Network API"
    icon_chr = chr(0xE307)
    icon_font = "pupil_icons"
    order = 0.01  # excecute first

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return cls.menu_label

    def __init__(self, g_pool, **kwargs):
        super().__init__(g_pool)

        # Frame Publisher setup
        self.__frame_publisher = FramePublisherController(**kwargs)
        self.__frame_publisher.add_observer(
            "on_format_changed", self.frame_publisher_announce_current_format
        )

        # Let existing eye-processes know about current frame publishing format
        self.frame_publisher_announce_current_format()

        # Pupil Remote setup
        self.__pupil_remote = PupilRemoteController(g_pool, **kwargs)
        self.__pupil_remote.add_observer(
            "on_pupil_remote_server_did_start", self.on_pupil_remote_server_did_start
        )
        self.__pupil_remote.add_observer(
            "on_pupil_remote_server_did_stop", self.on_pupil_remote_server_did_stop
        )

        # UI components setup (not initialized yet)
        self.__frame_publisher_ui_menu = FramePublisherMenu(self.__frame_publisher)
        self.__pupil_remote_ui_menu = PupilRemoteMenu(self.__pupil_remote)

    def get_init_dict(self):
        return {
            **self.__frame_publisher.get_init_dict(),
            **self.__pupil_remote.get_init_dict(),
        }

    def cleanup(self):
        self.frame_publisher_announce_stop()
        self.__frame_publisher = None
        self.__pupil_remote.cleanup()
        self.__pupil_remote = None

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.menu_label
        self.__pupil_remote_ui_menu.append_to_menu(self.menu)
        self.__frame_publisher_ui_menu.append_to_menu(self.menu)

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        frame = events.get("frame")
        if frame:
            world_frame_dicts = (
                self.__frame_publisher.create_world_frame_dicts_from_frame(frame)
            )
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
            # Let newly started eye-processes know about current frame publishing format
            self.frame_publisher_announce_current_format()
        elif notification["subject"] == "frame_publishing.set_format":
            # update format and trigger notification
            self.__frame_publisher.frame_format = notification["format"]

    def frame_publisher_announce_current_format(self, *_):
        self.notify_all(
            {
                "subject": "frame_publishing.started",
                "format": self.__frame_publisher.frame_format.value,
            }
        )

    def frame_publisher_announce_stop(self):
        self.notify_all({"subject": "frame_publishing.stopped"})

    def on_pupil_remote_server_did_start(self, address: str):
        pass

    def on_pupil_remote_server_did_stop(self):
        pass
