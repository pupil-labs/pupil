'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

if __name__ == '__main__':
    # make shared modules available across pupil_src
    from sys import path as syspath
    from os import path as ospath
    loc = ospath.abspath(__file__).rsplit('pupil_src', 1)
    syspath.append(ospath.join(loc[0], 'pupil_src', 'shared_modules'))
    del syspath, ospath

import os
from time import time
from glob import glob
from video_capture import init_playback_source, EndofVideoError
from player_methods import update_recording_to_recent, load_meta_info
from av_writer import AV_Writer
from file_methods import load_object
from player_methods import correlate_data, update_recording_to_recent


# logging
import logging

# Plug-ins
from plugin import Plugin_List, import_runtime_plugins
from vis_circle import Vis_Circle
from vis_cross import Vis_Cross
from vis_polyline import Vis_Polyline
from vis_light_points import Vis_Light_Points
from vis_watermark import Vis_Watermark
from vis_scan_path import Vis_Scan_Path
from vis_eye_video_overlay import Vis_Eye_Video_Overlay

# we are not importing manual gaze corrction. In Player corrections have already been applied.
# in batch exporter this plugin makes little sense.
from fixation_detector import Offline_Fixation_Detector


class Global_Container(object):
    pass


def export(rec_dir, user_dir, min_data_confidence, start_frame=None, end_frame=None,
           plugin_initializers=(), out_file_path=None, pre_computed={}):

    PID = str(os.getpid())
    logger = logging.getLogger(__name__+' with pid: '+PID)
    start_status = 'Starting video export with pid: {}'.format(PID)
    print(start_status)
    yield start_status, 0

    try:
        update_recording_to_recent(rec_dir)

        vis_plugins = sorted([Vis_Circle, Vis_Cross, Vis_Polyline, Vis_Light_Points,
                              Vis_Watermark, Vis_Scan_Path, Vis_Eye_Video_Overlay],
                             key=lambda x: x.__name__)
        analysis_plugins = [Offline_Fixation_Detector]
        user_plugins = sorted(import_runtime_plugins(os.path.join(user_dir, 'plugins')), key=lambda x: x.__name__)

        available_plugins = vis_plugins + analysis_plugins + user_plugins
        name_by_index = [p.__name__ for p in available_plugins]
        plugin_by_name = dict(zip(name_by_index, available_plugins))

        update_recording_to_recent(rec_dir)

        pupil_data_path = os.path.join(rec_dir, "pupil_data")
        audio_path = os.path.join(rec_dir, "audio.mp4")

        meta_info = load_meta_info(rec_dir)

        g_pool = Global_Container()
        g_pool.app = 'exporter'
        g_pool.min_data_confidence = min_data_confidence

        valid_ext = ('.mp4', '.mkv', '.avi', '.h264', '.mjpeg', '.fake')
        video_path = [f for f in glob(os.path.join(rec_dir, "world.*"))
                      if os.path.splitext(f)[1] in valid_ext][0]
        cap = init_playback_source(g_pool, source_path=video_path, timing=None)

        timestamps = cap.timestamps

        # Out file path verification, we do this before but if one uses a separate tool, this will kick in.
        if out_file_path is None:
            out_file_path = os.path.join(rec_dir, "world_viz.mp4")
        else:
            file_name = os.path.basename(out_file_path)
            dir_name = os.path.dirname(out_file_path)
            if not dir_name:
                dir_name = rec_dir
            if not file_name:
                file_name = 'world_viz.mp4'
            out_file_path = os.path.expanduser(os.path.join(dir_name, file_name))

        if os.path.isfile(out_file_path):
            logger.warning("Video out file already exsists. I will overwrite!")
            os.remove(out_file_path)
        logger.debug("Saving Video to {}".format(out_file_path))

        # Trim mark verification
        # make sure the trim marks (start frame, endframe) make sense:
        # We define them like python list slices, thus we can test them like such.
        trimmed_timestamps = timestamps[start_frame:end_frame]
        if len(trimmed_timestamps) == 0:
            warn = "Start and end frames are set such that no video will be exported."
            logger.warning(warn)
            yield warn, 0.
            return

        if start_frame is None:
            start_frame = 0

        # these two vars are shared with the lauching process and give a job length and progress report.
        frames_to_export = len(trimmed_timestamps)
        current_frame = 0
        exp_info = "Will export from frame {} to frame {}. This means I will export {} frames."
        logger.debug(exp_info.format(start_frame, start_frame + frames_to_export, frames_to_export))

        # setup of writer
        writer = AV_Writer(out_file_path, fps=cap.frame_rate, audio_loc=audio_path, use_timestamps=True)

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
        # load pupil_positions, gaze_positions
        pupil_data = pre_computed.get("pupil_data") or load_object(pupil_data_path)
        g_pool.pupil_data = pupil_data
        g_pool.pupil_positions = pre_computed.get("pupil_positions") or pupil_data['pupil_positions']
        g_pool.gaze_positions = pre_computed.get("gaze_positions") or pupil_data['gaze_positions']
        g_pool.fixations = pre_computed.get("fixations", [])

        g_pool.pupil_positions_by_frame = correlate_data(g_pool.pupil_positions, g_pool.timestamps)
        g_pool.gaze_positions_by_frame = correlate_data(g_pool.gaze_positions, g_pool.timestamps)
        g_pool.fixations_by_frame = correlate_data(g_pool.fixations, g_pool.timestamps)

        # add plugins
        g_pool.plugins = Plugin_List(g_pool, plugin_initializers)

        while frames_to_export > current_frame:
            try:
                frame = cap.get_frame()
            except EndofVideoError:
                break

            events = {'frame': frame}
            # new positons and events
            events['gaze_positions'] = g_pool.gaze_positions_by_frame[frame.index]
            events['pupil_positions'] = g_pool.pupil_positions_by_frame[frame.index]

            # publish delayed notifiactions when their time has come.
            for n in list(g_pool.delayed_notifications.values()):
                if n['_notify_time_'] < time():
                    del n['_notify_time_']
                    del g_pool.delayed_notifications[n['subject']]
                    g_pool.notifications.append(n)

            # notify each plugin if there are new notifactions:
            while g_pool.notifications:
                n = g_pool.notifications.pop(0)
                for p in g_pool.plugins:
                    p.on_notify(n)

            # allow each Plugin to do its work.
            for p in g_pool.plugins:
                p.recent_events(events)

            writer.write_video_frame(frame)
            current_frame += 1
            yield 'Exporting with pid {}'.format(PID), current_frame

        writer.close()
        writer = None

        duration = time()-start_time
        effective_fps = float(current_frame)/duration

        result = "Export done: Exported {} frames to {}. This took {} seconds. Exporter ran at {} frames per second."
        print(result.format(current_frame, out_file_path, duration, effective_fps))
        yield 'Export done. This took {:.0f} seconds.'.format(duration), current_frame

    except GeneratorExit:
        print('Video export with pid {} was canceled.'.format(os.getpid()))
    except Exception as e:
        from time import sleep
        import traceback
        trace = traceback.format_exc()
        print('Process Export (pid: {}) crashed with trace:\n{}'.format(os.getpid(), trace))
        yield e
        sleep(1.0)
