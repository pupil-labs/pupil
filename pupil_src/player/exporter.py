'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

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
import csv_utils
from time import time
from glob import glob
import cv2
import numpy as np
from video_capture import File_Source,EndofVideoFileError
from player_methods import correlate_data, is_pupil_rec_dir, update_recording_to_recent, load_meta_info
from methods import denormalize
from version_utils import VersionFormat, read_rec_version, get_version
from av_writer import AV_Writer
from file_methods import load_object

#logging
import logging

# Plug-ins
from plugin import Plugin_List,import_runtime_plugins
from vis_circle import Vis_Circle
from vis_cross import Vis_Cross
from vis_polyline import Vis_Polyline
from vis_light_points import Vis_Light_Points
from vis_watermark import Vis_Watermark

from scan_path import Scan_Path
from manual_gaze_correction import Manual_Gaze_Correction
from eye_video_overlay import Eye_Video_Overlay
from fixation_detector import Pupil_Angle_3D_Fixation_Detector,Gaze_Position_2D_Fixation_Detector


class Global_Container(object):
    pass

def export(should_terminate,frames_to_export,current_frame, rec_dir,user_dir,min_data_confidence,start_frame=None,end_frame=None,plugin_initializers=(),out_file_path=None):

    vis_plugins = sorted([Vis_Circle,Vis_Cross, Vis_Polyline,
        Vis_Light_Points, Vis_Watermark, Scan_Path], key=lambda x: x.__name__)
    analysis_plugins = sorted([Manual_Gaze_Correction,Eye_Video_Overlay,
        Pupil_Angle_3D_Fixation_Detector, Gaze_Position_2D_Fixation_Detector],
        key=lambda x: x.__name__)
    user_plugins = sorted(import_runtime_plugins(os.path.join(user_dir,'plugins')),
        key=lambda x: x.__name__)
    available_plugins = vis_plugins + analysis_plugins + user_plugins
    name_by_index = [p.__name__ for p in available_plugins]
    index_by_name = dict(zip(name_by_index,range(len(name_by_index))))
    plugin_by_name = dict(zip(name_by_index,available_plugins))

    logger = logging.getLogger(__name__+' with pid: '+str(os.getpid()) )

    update_recording_to_recent(rec_dir)

    video_path = [f for f in glob(os.path.join(rec_dir,"world.*")) if f[-3:] in ('mp4','mkv','avi')][0]
    timestamps_path = os.path.join(rec_dir, "world_timestamps.npy")
    pupil_data_path = os.path.join(rec_dir, "pupil_data")

    meta_info = load_meta_info(rec_dir)
    rec_version = read_rec_version(meta_info)

    g_pool = Global_Container()
    g_pool.app = 'exporter'
    g_pool.min_data_confidence = min_data_confidence
    timestamps = np.load(timestamps_path)
    cap = File_Source(g_pool,video_path,timestamps=timestamps)


    #Out file path verification, we do this before but if one uses a seperate tool, this will kick in.
    if out_file_path is None:
        out_file_path = os.path.join(rec_dir, "world_viz.mp4")
    else:
        file_name =  os.path.basename(out_file_path)
        dir_name = os.path.dirname(out_file_path)
        if not dir_name:
            dir_name = rec_dir
        if not file_name:
            file_name = 'world_viz.mp4'
        out_file_path = os.path.expanduser(os.path.join(dir_name,file_name))

    if os.path.isfile(out_file_path):
        logger.warning("Video out file already exsists. I will overwrite!")
        os.remove(out_file_path)
    logger.debug("Saving Video to %s"%out_file_path)


    #Trim mark verification
    #make sure the trim marks (start frame, endframe) make sense: We define them like python list slices,thus we can test them like such.
    trimmed_timestamps = timestamps[start_frame:end_frame]
    if len(trimmed_timestamps)==0:
        logger.warn("Start and end frames are set such that no video will be exported.")
        return False

    if start_frame == None:
        start_frame = 0

    #these two vars are shared with the lauching process and give a job length and progress report.
    frames_to_export.value = len(trimmed_timestamps)
    current_frame.value = 0
    logger.debug("Will export from frame %s to frame %s. This means I will export %s frames."%(start_frame,start_frame+frames_to_export.value,frames_to_export.value))

    #setup of writer
    writer = AV_Writer(out_file_path,fps=cap.frame_rate,use_timestamps=True)

    cap.seek_to_frame(start_frame)

    start_time = time()

    g_pool.capture = cap
    g_pool.rec_dir = rec_dir
    g_pool.user_dir = user_dir
    g_pool.rec_version = rec_version
    g_pool.timestamps = timestamps
    g_pool.delayed_notifications = {}
    g_pool.notifications = []


    # load pupil_positions, gaze_positions
    pupil_data = load_object(pupil_data_path)
    pupil_list = pupil_data['pupil_positions']
    gaze_list = pupil_data['gaze_positions']
    g_pool.pupil_positions_by_frame = correlate_data(pupil_list,g_pool.timestamps)
    g_pool.gaze_positions_by_frame = correlate_data(gaze_list,g_pool.timestamps)
    g_pool.fixations_by_frame = [[] for x in g_pool.timestamps] #populated by the fixation detector plugin

    #add plugins
    g_pool.plugins = Plugin_List(g_pool,plugin_by_name,plugin_initializers)

    while frames_to_export.value > current_frame.value:

        if should_terminate.value:
            logger.warning("User aborted export. Exported %s frames to %s."%(current_frame.value,out_file_path))

            #explicit release of VideoWriter
            writer.close()
            writer = None
            return False

        try:
            frame = cap.get_frame_nowait()
        except EndofVideoFileError:
            break

        events = {}
        #new positons and events
        events['gaze_positions'] = g_pool.gaze_positions_by_frame[frame.index]
        events['pupil_positions'] = g_pool.pupil_positions_by_frame[frame.index]

        # publish delayed notifiactions when their time has come.
        for n in g_pool.delayed_notifications.values():
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
            p.update(frame,events)

        writer.write_video_frame(frame)
        current_frame.value +=1

    writer.close()
    writer = None

    duration = time()-start_time
    effective_fps = float(current_frame.value)/duration

    logger.info("Export done: Exported %s frames to %s. This took %s seconds. Exporter ran at %s frames per second"%(current_frame.value,out_file_path,duration,effective_fps))
    return True
