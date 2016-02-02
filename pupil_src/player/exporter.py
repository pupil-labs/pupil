'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
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
from glob import glob
import cv2
import numpy as np
from video_capture import File_Capture,EndofVideoFileError
from player_methods import correlate_data,update_recording_0v4_to_current,update_recording_0v3_to_current
from methods import denormalize
from version_utils import VersionFormat, read_rec_version, get_version
from av_writer import AV_Writer
from file_methods import load_object

#logging
import logging

# Plug-ins
from plugin import Plugin_List
from vis_circle import Vis_Circle
from vis_cross import Vis_Cross
from vis_polyline import Vis_Polyline
from vis_light_points import Vis_Light_Points
from vis_watermark import Vis_Watermark

from scan_path import Scan_Path
from manual_gaze_correction import Manual_Gaze_Correction
from eye_video_overlay import Eye_Video_Overlay
from fixation_detector import Dispersion_Duration_Fixation_Detector


available_plugins = Vis_Circle,Vis_Cross, Vis_Polyline, \
                    Vis_Light_Points, Vis_Watermark, \
                    Scan_Path, \
                    Manual_Gaze_Correction,Eye_Video_Overlay, \
                    Dispersion_Duration_Fixation_Detector
name_by_index = [p.__name__ for p in available_plugins]
index_by_name = dict(zip(name_by_index,range(len(name_by_index))))
plugin_by_name = dict(zip(name_by_index,available_plugins))

class Global_Container(object):
        pass

def export(should_terminate,frames_to_export,current_frame, rec_dir,user_dir,start_frame=None,end_frame=None,plugin_initializers=[],out_file_path=None):

    logger = logging.getLogger(__name__+' with pid: '+str(os.getpid()) )

   #parse info.csv file
    meta_info_path = os.path.join(rec_dir,"info.csv")
    with open(meta_info_path) as info:
        meta_info = dict( ((line.strip().split('\t')) for line in info.readlines() ) )

    video_path = glob(os.path.join(rec_dir,"world.*"))[0]
    timestamps_path = os.path.join(rec_dir, "world_timestamps.npy")
    pupil_data_path = os.path.join(rec_dir, "pupil_data")


    rec_version = read_rec_version(meta_info)
    if rec_version >= VersionFormat('0.5'):
        pass
    elif rec_version >= VersionFormat('0.4'):
        update_recording_0v4_to_current(rec_dir)
    elif rec_version >= VersionFormat('0.3'):
        update_recording_0v3_to_current(rec_dir)
        timestamps_path = os.path.join(rec_dir, "timestamps.npy")
    else:
        logger.Error("This recording is to old. Sorry.")
        return


    timestamps = np.load(timestamps_path)

    cap = File_Capture(video_path,timestamps=timestamps)


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

    g = Global_Container()
    g.app = 'exporter'
    g.capture = cap
    g.rec_dir = rec_dir
    g.user_dir = user_dir
    g.rec_version = rec_version
    g.timestamps = timestamps


    # load pupil_positions, gaze_positions
    pupil_data = load_object(pupil_data_path)
    pupil_list = pupil_data['pupil_positions']
    gaze_list = pupil_data['gaze_positions']

    g.pupil_positions_by_frame = correlate_data(pupil_list,g.timestamps)
    g.gaze_positions_by_frame = correlate_data(gaze_list,g.timestamps)
    g.fixations_by_frame = [[] for x in g.timestamps] #populated by the fixation detector plugin

    #add plugins
    g.plugins = Plugin_List(g,plugin_by_name,plugin_initializers)

    while frames_to_export.value - current_frame.value > 0:

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
        events['gaze_positions'] = g.gaze_positions_by_frame[frame.index]
        events['pupil_positions'] = g.pupil_positions_by_frame[frame.index]

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
