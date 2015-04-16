'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
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
import cv2
import numpy as np
from video_capture import autoCreateCapture,EndofVideoFileError
from player_methods import correlate_gaze,correlate_gaze_legacy
from methods import denormalize
from version_utils import VersionFormat, read_rec_version, get_version
from av_writer import AV_Writer
#logging
import logging

# Plug-ins
from plugin import Plugin_List
from vis_circle import Vis_Circle
from vis_cross import Vis_Cross
from vis_polyline import Vis_Polyline
from display_gaze import Display_Gaze
from vis_light_points import Vis_Light_Points
from vis_watermark import Vis_Watermark

from scan_path import Scan_Path
from filter_fixations import Filter_Fixations
from manual_gaze_correction import Manual_Gaze_Correction
from eye_video_overlay import Eye_Video_Overlay

available_plugins =  Vis_Circle,Vis_Cross, Vis_Polyline, Vis_Light_Points, Vis_Watermark, Scan_Path,Filter_Fixations,Manual_Gaze_Correction,Eye_Video_Overlay
name_by_index = [p.__name__ for p in available_plugins]
index_by_name = dict(zip(name_by_index,range(len(name_by_index))))
plugin_by_name = dict(zip(name_by_index,available_plugins))

class Global_Container(object):
        pass

def export(should_terminate,frames_to_export,current_frame, rec_dir,user_dir,start_frame=None,end_frame=None,plugin_initializers=[],out_file_path=None):

    logger = logging.getLogger(__name__+' with pid: '+str(os.getpid()) )



    #parse info.csv file
    with open(rec_dir + "/info.csv") as info:
        meta_info = dict( ((line.strip().split('\t')) for line in info.readlines() ) )
    rec_version = read_rec_version(meta_info)
    logger.debug("Exporting a video from recording with version: %s"%rec_version)

    if rec_version < VersionFormat('0.4'):
        video_path = rec_dir + "/world.avi"
        timestamps_path = rec_dir + "/timestamps.npy"
    else:
        video_path = rec_dir + "/world.mkv"
        timestamps_path = rec_dir + "/world_timestamps.npy"

    gaze_positions_path = rec_dir + "/gaze_positions.npy"
    #load gaze information
    gaze_list = np.load(gaze_positions_path)
    timestamps = np.load(timestamps_path)

    #correlate data
    if rec_version < VersionFormat('0.4'):
        positions_by_frame = correlate_gaze_legacy(gaze_list,timestamps)
    else:
        positions_by_frame = correlate_gaze(gaze_list,timestamps)

    cap = autoCreateCapture(video_path,timestamps=timestamps_path)
    width,height = cap.frame_size

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
    writer = AV_Writer(out_file_path)

    cap.seek_to_frame(start_frame)

    start_time = time()

    g = Global_Container()
    g.app = 'exporter'
    g.rec_dir = rec_dir
    g.user_dir = user_dir
    g.rec_version = rec_version
    g.timestamps = timestamps
    g.gaze_list = gaze_list
    g.positions_by_frame = positions_by_frame
    g.plugins = Plugin_List(g,plugin_by_name,plugin_initializers)

    while frames_to_export.value - current_frame.value > 0:

        if should_terminate.value:
            logger.warning("User aborted export. Exported %s frames to %s."%(current_frame.value,out_file_path))

            #explicit release of VideoWriter
            writer.close()
            writer = None
            return False

        try:
            frame = cap.get_frame()
        except EndofVideoFileError:
            break

        events = {}
        #new positons and events
        events['pupil_positions'] = positions_by_frame[frame.index]
        # allow each Plugin to do its work.
        for p in g.plugins:
            p.update(frame,events)

        writer.write_video_frame(frame)
        current_frame.value +=1

    writer.close()
    writer = None

    duration = time()-start_time
    effective_fps = float(current_frame.value)/duration

    logger.info("Export done: Exported %s frames to %s. This took %s seconds. Exporter ran at %s frames per second"%(current_frame.value,out_file_path,duration,effective_fps))
    return True
