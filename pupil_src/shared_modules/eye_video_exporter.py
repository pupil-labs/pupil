"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import csv
import logging
import os
from shutil import copy2

import player_methods as pm
from methods import denormalize
from video_exporter import VideoExporter

logger = logging.getLogger(__name__)


def _process_frame(capture, frame):
    return frame.img  # change nothing


class Eye_Video_Exporter(VideoExporter):
    icon_chr = "EV"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        logger.info("Eye Video Exporter has been launched.")

    def customize_menu(self):
        self.menu.label = "Eye Video Exporter"

    def export_data(self, export_range, export_dir):
        self.add_export_job(
            export_range, export_dir, "EyeVideo", "eye0", "eye0", _process_frame
        )
        self.add_export_job(
            export_range, export_dir, "EyeVideo", "eye1", "eye1", _process_frame
        )
