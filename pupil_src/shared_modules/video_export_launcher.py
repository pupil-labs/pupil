"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import os
import time

from pyglui import ui

import background_helper as bh
import player_methods as pm
from exporter import export
from plugin import Analysis_Plugin_Base

logger = logging.getLogger(__name__)


def verify_out_file_path(out_file_path, rec_dir):
    # Out file path verification
    if not out_file_path:
        out_file_path = os.path.join(rec_dir, "world.mp4")
    else:
        file_name = os.path.basename(out_file_path)
        dir_name = os.path.dirname(out_file_path)
        if not dir_name:
            dir_name = rec_dir
        if not file_name:
            file_name = "world.mp4"
        out_file_path = os.path.expanduser(os.path.join(dir_name, file_name))

    if os.path.isfile(out_file_path):
        logger.warning("Video out file already exsists. I will overwrite!")
        os.remove(out_file_path)
    logger.debug("Saving Video to {}".format(out_file_path))

    return out_file_path


class Video_Export_Launcher(Analysis_Plugin_Base):
    """docstring for Video_Export_Launcher
    this plugin can export the video in a seperate process using exporter
    """

    icon_chr = chr(0xec09)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        # initialize empty menu
        self.exports = []
        self.rec_name = "world.mp4"

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Export Video"
        # add menu to the window
        self._update_ui()

    def _update_ui(self):
        del self.menu.elements[:]
        self.menu.append(
            ui.Info_Text(
                "The export can be found in the exports folder, under the recording directory."
            )
        )
        self.menu.append(
            ui.Info_Text(
                "Select your export frame range using the trim marks in the seek bar. This will affect all exporting plugins."
            )
        )
        self.menu.append(
            ui.Info_Text("Press the export button or type 'e' to start the export.")
        )

        for job in self.exports[::-1]:
            submenu = ui.Growing_Menu(job.out_file_path)
            submenu.append(
                ui.Text_Input("status", job, label="Status", setter=lambda x: None)
            )
            progress_bar = ui.Slider(
                "progress", job, min=0, max=job.frames_to_export, label="Progress"
            )
            progress_bar.read_only = True
            submenu.append(progress_bar)
            submenu.append(ui.Button("Cancel", job.cancel))
            self.menu.append(submenu)

    def deinit_ui(self):
        self.remove_menu()

    def on_notify(self, notification):
        if notification["subject"] == "should_export":
            self.add_export(notification["range"], notification["export_dir"])

    def add_export(self, export_range, export_dir):
        logger.warning("Adding new video export process.")

        rec_dir = self.g_pool.rec_dir
        user_dir = self.g_pool.user_dir
        # export_range.stop is exclusive
        start_frame, end_frame = export_range

        # Here we make clones of every plugin that supports it.
        # So it runs in the current config when we lauch the exporter.
        plugins = self.g_pool.plugins.get_initializers()

        out_file_path = verify_out_file_path(self.rec_name, export_dir)
        pre_computed = self.precomputed_for_range(export_range)

        args = (
            rec_dir,
            user_dir,
            self.g_pool.min_data_confidence,
            start_frame,
            end_frame,
            plugins,
            out_file_path,
            pre_computed,
        )
        process = bh.Task_Proxy(
            "Pupil Export {}".format(out_file_path), export, args=args
        )
        process.out_file_path = out_file_path
        process.frames_to_export = end_frame - start_frame
        process.status = ""
        process.progress = 0
        self.exports.append(process)
        logger.debug("Starting export as new process {}".format(process))
        self._update_ui()

    def recent_events(self, events):
        for e in self.exports:
            try:
                recent = [d for d in e.fetch()]
            except Exception as e:
                self.status, self.progress = "{}: {}".format(type(e).__name__, e), 0
            else:
                if recent:
                    e.status, e.progress = recent[-1]
                if e.canceled:
                    e.status = "Export has been canceled."

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        for e in self.exports:
            e.cancel()

    def precomputed_for_range(self, export_range):
        export_window = pm.exact_window(self.g_pool.timestamps, export_range)
        pre_computed = {
            "gaze": self.g_pool.gaze_positions,
            "pupil": self.g_pool.pupil_positions,
            "fixations": self.g_pool.fixations,
        }

        for key, bisector in pre_computed.items():
            init_dict = bisector.init_dict_for_window(export_window)
            init_dict["data"] = [datum.serialized for datum in init_dict["data"]]
            pre_computed[key] = init_dict

        return pre_computed

