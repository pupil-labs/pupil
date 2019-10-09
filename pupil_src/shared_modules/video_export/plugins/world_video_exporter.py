"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import os

import player_methods as pm
from task_manager import ManagedTask
from video_export.plugin_base.video_exporter import VideoExporter
from pupil_recording import PupilRecording

logger = logging.getLogger(__name__)


class World_Video_Exporter(VideoExporter):
    """
    Exports the world video as seen in Player (i.e. including all plugin renderings).
    """

    icon_chr = chr(0xEC09)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool, max_concurrent_tasks=1)
        self.logger = logging.getLogger(__name__)
        self.logger.info("World Video Exporter has been launched.")
        self.rec_name = "world.mp4"

    def customize_menu(self):
        self.menu.label = "World Video Exporter"
        super().customize_menu()

    def export_data(self, export_range, export_dir):
        rec_dir = self.g_pool.rec_dir
        user_dir = self.g_pool.user_dir
        start_frame, end_frame = export_range

        # Here we make clones of every plugin that supports it.
        # So it runs in the current config when we launch the exporter.
        plugins = self.g_pool.plugins.get_initializers()

        out_file_path = os.path.join(export_dir, self.rec_name)
        pre_computed_eye_data = self._precomputed_eye_data_for_range(export_range)

        args = (
            rec_dir,
            user_dir,
            self.g_pool.min_data_confidence,
            start_frame,
            end_frame,
            plugins,
            out_file_path,
            pre_computed_eye_data,
        )
        task = ManagedTask(
            _export_world_video,
            args=args,
            heading="Export World Video",
            min_progress=0.0,
            max_progress=end_frame - start_frame,
        )
        self.add_task(task)

    def _precomputed_eye_data_for_range(self, export_range):
        export_window = pm.exact_window(self.g_pool.timestamps, export_range)
        pre_computed = {
            "gaze": self.g_pool.gaze_positions,
            "pupil": self.g_pool.pupil_positions,
            "pupil_by_id_0": self.g_pool.pupil_positions_by_id[0],
            "pupil_by_id_1": self.g_pool.pupil_positions_by_id[1],
            "fixations": self.g_pool.fixations,
        }

        for key, bisector in pre_computed.items():
            init_dict = bisector.init_dict_for_window(export_window)
            init_dict["data"] = [datum.serialized for datum in init_dict["data"]]
            pre_computed[key] = init_dict

        return pre_computed


class GlobalContainer(object):
    pass


def _export_world_video(
    rec_dir,
    user_dir,
    min_data_confidence,
    start_frame,
    end_frame,
    plugin_initializers,
    out_file_path,
    pre_computed_eye_data,
):
    """
    Simulates the generation for the world video and saves a certain time range as a video.
    It simulates a whole g_pool such that all plugins run as normal.
    """
    from glob import glob
    from time import time

    import file_methods as fm
    import player_methods as pm
    from av_writer import MPEG_Audio_Writer

    # we are not importing manual gaze correction. In Player corrections have already been applied.
    # in batch exporter this plugin makes little sense.
    from fixation_detector import Offline_Fixation_Detector
    from eye_movement import Offline_Eye_Movement_Detector

    # Plug-ins
    from plugin import Plugin_List, import_runtime_plugins
    from video_capture import EndofVideoError, File_Source
    from video_overlay.plugins import Video_Overlay, Eye_Overlay
    from vis_circle import Vis_Circle
    from vis_cross import Vis_Cross
    from vis_light_points import Vis_Light_Points
    from vis_polyline import Vis_Polyline
    from vis_scan_path import Vis_Scan_Path
    from vis_watermark import Vis_Watermark

    PID = str(os.getpid())
    logger = logging.getLogger(__name__ + " with pid: " + PID)
    start_status = "Starting video export with pid: {}".format(PID)
    logger.info(start_status)
    yield start_status, 0

    try:
        vis_plugins = sorted(
            [
                Vis_Circle,
                Vis_Cross,
                Vis_Polyline,
                Vis_Light_Points,
                Vis_Watermark,
                Vis_Scan_Path,
                Eye_Overlay,
                Video_Overlay,
            ],
            key=lambda x: x.__name__,
        )
        analysis_plugins = [Offline_Fixation_Detector, Offline_Eye_Movement_Detector]
        user_plugins = sorted(
            import_runtime_plugins(os.path.join(user_dir, "plugins")),
            key=lambda x: x.__name__,
        )

        available_plugins = vis_plugins + analysis_plugins + user_plugins
        name_by_index = [p.__name__ for p in available_plugins]
        plugin_by_name = dict(zip(name_by_index, available_plugins))

        recording = PupilRecording(rec_dir)
        meta_info = recording.meta_info

        g_pool = GlobalContainer()
        g_pool.app = "exporter"
        g_pool.min_data_confidence = min_data_confidence

        videos = recording.files().core().world().videos()
        if not videos:
            raise FileNotFoundError("No world video found")

        source_path = videos[0].resolve()
        cap = File_Source(g_pool, source_path=source_path, fill_gaps=True, timing=None)
        if not cap.initialised:
            warn = "Trying to export zero-duration world video."
            logger.warning(warn)
            yield warn, 0.0
            return

        timestamps = cap.timestamps

        file_name = os.path.basename(out_file_path)
        dir_name = os.path.dirname(out_file_path)
        out_file_path = os.path.expanduser(os.path.join(dir_name, file_name))

        if os.path.isfile(out_file_path):
            logger.warning("Video out file already exsists. I will overwrite!")
            os.remove(out_file_path)
        logger.debug("Saving Video to {}".format(out_file_path))

        # Trim mark verification
        # make sure the trim marks (start frame, end frame) make sense:
        # We define them like python list slices, thus we can test them like such.
        trimmed_timestamps = timestamps[start_frame:end_frame]
        if len(trimmed_timestamps) == 0:
            warn = "Start and end frames are set such that no video will be exported."
            logger.warning(warn)
            yield warn, 0.0
            return

        if start_frame is None:
            start_frame = 0

        # these two vars are shared with the launching process and give a job length and progress report.
        frames_to_export = len(trimmed_timestamps)
        current_frame = 0
        exp_info = (
            "Will export from frame {} to frame {}. This means I will export {} frames."
        )
        logger.debug(
            exp_info.format(
                start_frame, start_frame + frames_to_export, frames_to_export
            )
        )

        # setup of writer
        writer = MPEG_Audio_Writer(
            out_file_path, start_time_synced=trimmed_timestamps[0], audio_dir=rec_dir
        )

        cap.seek_to_frame(start_frame)

        start_time = time()

        g_pool.plugin_by_name = plugin_by_name
        g_pool.capture = cap
        g_pool.rec_dir = rec_dir
        g_pool.user_dir = user_dir
        g_pool.meta_info = meta_info
        g_pool.timestamps = timestamps
        g_pool.delayed_notifications = {}
        g_pool.notifications = []

        for initializers in pre_computed_eye_data.values():
            initializers["data"] = [
                fm.Serialized_Dict(msgpack_bytes=serialized)
                for serialized in initializers["data"]
            ]

        g_pool.pupil_positions = pm.Bisector(**pre_computed_eye_data["pupil"])
        g_pool.pupil_positions_by_id = (
            pm.Bisector(**pre_computed_eye_data["pupil_by_id_0"]),
            pm.Bisector(**pre_computed_eye_data["pupil_by_id_1"]),
        )
        g_pool.gaze_positions = pm.Bisector(**pre_computed_eye_data["gaze"])
        g_pool.fixations = pm.Affiliator(**pre_computed_eye_data["fixations"])

        # add plugins
        g_pool.plugins = Plugin_List(g_pool, plugin_initializers)

        while frames_to_export > current_frame:
            try:
                frame = cap.get_frame()
            except EndofVideoError:
                break

            events = {"frame": frame}
            # new positions and events
            frame_window = pm.enclosing_window(g_pool.timestamps, frame.index)
            events["gaze"] = g_pool.gaze_positions.by_ts_window(frame_window)
            events["pupil"] = g_pool.pupil_positions.by_ts_window(frame_window)

            # publish delayed notifications when their time has come.
            for n in list(g_pool.delayed_notifications.values()):
                if n["_notify_time_"] < time():
                    del n["_notify_time_"]
                    del g_pool.delayed_notifications[n["subject"]]
                    g_pool.notifications.append(n)

            # notify each plugin if there are new notifications:
            while g_pool.notifications:
                n = g_pool.notifications.pop(0)
                for p in g_pool.plugins:
                    p.on_notify(n)

            # allow each Plugin to do its work.
            for p in g_pool.plugins:
                p.recent_events(events)

            writer.write_video_frame(frame)
            current_frame += 1
            yield "Exporting with pid {}".format(PID), current_frame

        writer.close(timestamp_export_format="all")

        duration = time() - start_time
        effective_fps = float(current_frame) / duration

        result = "Export done: Exported {} frames to {}. This took {} seconds. Exporter ran at {} frames per second."
        logger.info(
            result.format(current_frame, out_file_path, duration, effective_fps)
        )
        yield "Export done. This took {:.0f} seconds.".format(duration), current_frame

    except GeneratorExit:
        logger.warning("Video export with pid {} was canceled.".format(os.getpid()))
