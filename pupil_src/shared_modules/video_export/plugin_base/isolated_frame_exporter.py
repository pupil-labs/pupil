"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import os
from abc import ABCMeta
from glob import glob

import player_methods as pm
from av_writer import AV_Writer
from task_manager import ManagedTask
from video_capture import File_Source, EndofVideoError
from video_export.plugin_base.video_exporter import VideoExporter


class IsolatedFrameExporter(VideoExporter, metaclass=ABCMeta):
    """
    A VideoExporter that exports a part or all of some video file and applies
    a function process_frame to every frame.
    """

    def add_export_job(
        self,
        export_range,
        export_dir,
        input_name,
        output_name,
        process_frame,
        timestamp_export_format,
    ):
        os.makedirs(export_dir, exist_ok=True)
        self.logger.info("Exporting to {}".format(export_dir))

        input_video_file = _find_video_file(self.g_pool.rec_dir, input_name)
        output_video_file = os.path.join(export_dir, output_name + ".mp4")
        task_args = (
            input_video_file,
            output_video_file,
            export_range,
            self.g_pool.timestamps,
            process_frame,
            timestamp_export_format,
        )
        task = ManagedTask(
            _convert_video_file,
            args=task_args,
            heading="Export {} Video".format(input_name),
            min_progress=0.0,
            max_progress=100.0,
        )
        self.add_task(task)


def _find_video_file(directory, name):
    try:
        return next(
            f
            for f in glob(os.path.join(directory, name + ".*"))
            if os.path.splitext(f)[-1] in (".mp4", ".mkv", ".avi", ".mjpeg")
        )
    except StopIteration:
        raise FileNotFoundError("No Video " + name + " found")


def _convert_video_file(
    input_file,
    output_file,
    export_range,
    world_timestamps,
    process_frame,
    timestamp_export_format,
):
    yield "Export video", 0.0
    input_source = File_Source(EmptyGPool(), input_file)
    if not input_source.initialised:
        yield "Exporting video failed", 0.0
        return

    # yield progress results two times per second
    update_rate = int(input_source.frame_rate / 2)

    export_window = pm.exact_window(world_timestamps, export_range)
    (export_from_index, export_to_index) = pm.find_closest(
        input_source.timestamps, export_window
    )
    writer = AV_Writer(
        output_file, fps=input_source.frame_rate, audio_loc=None, use_timestamps=True
    )
    input_source.seek_to_frame(export_from_index)
    next_update_idx = export_from_index + update_rate
    while True:
        try:
            input_frame = input_source.get_frame()
        except EndofVideoError:
            break

        if input_frame.index > export_to_index:
            break

        output_img = process_frame(input_source, input_frame)
        output_frame = input_frame
        output_frame._img = output_img  # it's ._img because .img has no setter
        writer.write_video_frame(output_frame)

        if input_source.current_frame_idx >= next_update_idx:
            progress = (input_source.current_frame_idx - export_from_index) / (
                export_to_index - export_from_index
            )
            yield "Exporting video", progress * 100.0
            next_update_idx += update_rate

    writer.close(timestamp_export_format)
    input_source.cleanup()
    yield "Exporting video completed", 100.0


class EmptyGPool:
    pass
