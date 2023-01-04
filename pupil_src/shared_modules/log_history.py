"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

# logging
import logging
import os

import zmq_tools
from plugin import Plugin
from pyglui import ui

logger = logging.getLogger(__name__)


class Log_to_Callback(logging.Handler):
    def __init__(self, cb):
        super().__init__()
        self.cb = cb

    def emit(self, record):
        self.cb(record)


class Log_History(Plugin):
    """Simple logging GUI that displays the last N messages from the logger"""

    icon_chr = chr(0xEC10)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.menu = None
        self.num_messages = 50

        self.formatter = logging.Formatter(
            "%(processName)s - [%(levelname)s] %(name)s: %(message)s"
        )
        self.logfile = os.path.join(self.g_pool.user_dir, self.g_pool.app + ".log")

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Log"

        help_str = 'A View of the {} most recent log messages. Complete logs are here: "{}"'.format(
            self.num_messages, self.g_pool.user_dir
        )
        self.menu.append(ui.Info_Text(help_str))

        with open(self.logfile, encoding="utf-8") as fh:
            for l in fh.readlines():
                self.menu.insert(2, ui.Info_Text(l[26:-1]))

        if self.g_pool.app == "capture":
            self.log_handler = None
            self._socket = zmq_tools.Msg_Receiver(
                self.g_pool.zmq_ctx, self.g_pool.ipc_sub_url, ("logging",)
            )

        else:
            self._socket = None
            self.log_handler = Log_to_Callback(self.on_log)
            logger = logging.getLogger()
            logger.addHandler(self.log_handler)
            self.log_handler.setLevel(logging.INFO)

    def recent_events(self, events):
        if self._socket:
            while self._socket.new_data:
                t, s = self._socket.recv()
                self.on_log(logging.makeLogRecord(s))

    def on_log(self, record):
        self.menu.elements[self.num_messages + 2 :] = []
        self.menu.insert(1, ui.Info_Text(str(self.formatter.format(record))))

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        if self.log_handler:
            logger = logging.getLogger()
            logger.removeHandler(self.log_handler)
        if self._socket:
            del self._socket

    def get_init_dict(self):
        return {}
