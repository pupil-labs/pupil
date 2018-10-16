"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import abc
import logging
import multiprocessing as mp
import os
from fractions import Fraction
from glob import glob

import av
from pyglui import ui

import numpy as np
import background_helper as bh
import csv_utils
import player_methods as pm
from plugin import Analysis_Plugin_Base
from video_capture import File_Source, EndofVideoError

logger = logging.getLogger(__name__)


class Empty(object):
    pass


def write_timestamps(file_loc, timestamps):
    directory, video_file = os.path.split(file_loc)
    name, ext = os.path.splitext(video_file)
    ts_file = "{}_timestamps.npy".format(name)
    ts_loc = os.path.join(directory, ts_file)
    ts = np.array(timestamps)
    np.save(ts_loc, ts)


def export_processed_h264(
    world_timestamps,
    unprocessed_video_loc,
    target_video_loc,
    export_range,
    process_frame,
    export_timestamps,
):
    yield "Converting video", .1
    capture = File_Source(Empty(), unprocessed_video_loc)
    if not capture.initialised:
        yield "Converting scene video failed", 0.
        return

    export_window = pm.exact_window(world_timestamps, export_range)
    (export_from_index, export_to_index) = pm.find_closest(
        capture.timestamps, export_window
    )

    update_rate = 10
    start_time = None
    time_base = Fraction(1, 65535)

    target_container = av.open(target_video_loc, "w")
    video_stream = target_container.add_stream("mpeg4", 1 / time_base)
    video_stream.bit_rate = 150e6
    video_stream.bit_rate_tolerance = video_stream.bit_rate / 20
    video_stream.thread_count = max(1, mp.cpu_count() - 1)
    video_stream.width, video_stream.height = capture.frame_size

    av_frame = av.VideoFrame(*capture.frame_size, "bgr24")
    av_frame.time_base = time_base

    capture.seek_to_frame(export_from_index)
    next_update_idx = export_from_index + update_rate
    timestamps = []
    while True:
        try:
            frame = capture.get_frame()
        except EndofVideoError:
            break

        if frame.index > export_to_index:
            break

        if start_time is None:
            start_time = frame.timestamp

        undistorted_img = process_frame(capture, frame)
        av_frame.planes[0].update(undistorted_img)
        av_frame.pts = int((frame.timestamp - start_time) / time_base)

        if export_timestamps:
            timestamps.append(frame.timestamp)

        packet = video_stream.encode(av_frame)
        if packet:
            target_container.mux(packet)

        if capture.current_frame_idx >= next_update_idx:
            progress = (
                (capture.current_frame_idx - export_from_index)
                / (export_to_index - export_from_index)
            ) * .9 + .1
            yield "Converting video", progress * 100.
            next_update_idx += update_rate

    while True:  # flush encoder
        packet = video_stream.encode()
        if packet:
            target_container.mux(packet)
        else:
            break

    if export_timestamps:
        write_timestamps(target_video_loc, timestamps)

    target_container.close()
    capture.cleanup()
    yield "Converting video completed", 1. * 100.


class VideoExporter(Analysis_Plugin_Base):
    """Base class for the iMotions and Eye Video Export plugins.

    Supports the export of one or more videos.
    You need to override customize_menu() and export_data().
    Call add_export_job() to create a new video export task.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.export_tasks = []
        self.status = "Not exporting"
        self.progress = 0.
        self.output = "Not set yet"

    def init_ui(self):
        self.add_menu()
        self.customize_menu()
        self.menu.append(
            ui.Text_Input("status", self, label="Status", setter=lambda _: None)
        )
        self.menu.append(
            ui.Text_Input("output", self, label="Last export", setter=lambda _: None)
        )
        self.menu.append(ui.Slider("progress", self, label="Progress"))
        self.menu[-1].read_only = True
        self.menu[-1].display_format = "%.0f%%"
        self.menu.append(ui.Button("Cancel export", self.cancel))

    @abc.abstractmethod
    def customize_menu(self):
        pass

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        self.cancel()

    def on_notify(self, notification):
        if notification["subject"] == "should_export":
            self.cancel()
            self._start_export(notification["range"], notification["export_dir"])

    def recent_events(self, events):
        self._update_task_progress()

    def _update_task_progress(self):
        num_considered_tasks = 0
        total_progress = 0.0
        for task in self.export_tasks:
            recent = [d for d in task.fetch()]
            if recent:
                status, progress = recent[-1]
                # even with multiple tasks, each status is currently "Export video"
                # so we don't need to merge the different status values (for now)
                self.status = status
                total_progress += progress
                num_considered_tasks += 1
        if num_considered_tasks > 0:
            progress = total_progress / num_considered_tasks
            # with multiple tasks the progress usually jumps because we often
            # don't get a progress from each task
            if progress > self.progress:
                self.progress = progress

    def gl_display(self):
        self.menu_icon.indicator_stop = self.progress / 100.

    def _start_export(self, export_range, export_dir):
        self.progress = 0.0
        self.export_data(export_range, export_dir)

    @abc.abstractmethod
    def export_data(self, export_range, export_dir):
        pass

    def add_export_job(
        self,
        export_range,
        export_dir,
        plugin_name,
        input_name,
        output_name,
        process_frame,
        export_timestamps,
    ):
        os.makedirs(export_dir, exist_ok=True)
        logger.info("Exporting to {}".format(export_dir))

        try:
            distorted_video_loc = [
                f
                for f in glob(os.path.join(self.g_pool.rec_dir, input_name + ".*"))
                if os.path.splitext(f)[-1] in (".mp4", ".mkv", ".avi", ".mjpeg")
            ][0]
        except IndexError:
            raise FileNotFoundError("No Video " + input_name + " found")

        target_video_loc = os.path.join(export_dir, output_name + ".mp4")
        generator_args = (
            self.g_pool.timestamps,
            distorted_video_loc,
            target_video_loc,
            export_range,
            process_frame,
            export_timestamps,
        )
        task = bh.Task_Proxy(
            plugin_name + " Video Export", export_processed_h264, args=generator_args
        )
        self.export_tasks.append(task)
        return {"export_folder": export_dir}

    def cancel(self):
        for task in self.export_tasks:
            task.cancel()
        self.export_tasks = []
        self.status = "Export has been canceled"
        self.progress = 0.0

    def _get_recording_start_date(self):
        csv_loc = os.path.join(self.g_pool.rec_dir, "info.csv")
        with open(csv_loc, "r", encoding="utf-8") as csvfile:
            rec_info = csv_utils.read_key_value_file(csvfile)
            date = rec_info["Start Date"].replace(".", "_").replace(":", "_")
            time = rec_info["Start Time"].replace(":", "_")
        return "{}_{}".format(date, time)
