'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

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
from uvc_capture import autoCreateCapture
from player_methods import correlate_gaze
from methods import denormalize, Temp
#logging
import logging

# Plug-ins
from vis_circle import Vis_Circle
from vis_cross import Vis_Cross
from vis_polyline import Vis_Polyline
from vis_light_points import Vis_Light_Points
from scan_path import Scan_Path
from filter_fixations import Filter_Fixations
from manual_gaze_correction import Manual_Gaze_Correction

plugin_by_index =  (  Vis_Circle,
                        Vis_Cross,
                        Vis_Polyline,
                        Scan_Path,
                        Vis_Light_Points,
                        Filter_Fixations,
                        Manual_Gaze_Correction
                        )

name_by_index = [p.__name__ for p in plugin_by_index]
index_by_name = dict(zip(name_by_index,range(len(name_by_index))))
plugin_by_name = dict(zip(name_by_index,plugin_by_index))



def export(should_terminate,frames_to_export,current_frame, data_dir,start_frame=None,end_frame=None,plugin_initializers=[],out_file_path=None):

    logger = logging.getLogger(__name__+' with pid: '+str(os.getpid()) )

    #parse and load data dir info
    video_path = data_dir + "/world.avi"
    timestamps_path = data_dir + "/timestamps.npy"
    gaze_positions_path = data_dir + "/gaze_positions.npy"
    record_path = data_dir + "/world_viz.avi"


    #parse info.csv file
    with open(data_dir + "/info.csv") as info:
        meta_info = dict( ((line.strip().split('\t')) for line in info.readlines() ) )
    rec_version = meta_info["Capture Software Version"]
    rec_version_int = int(filter(type(rec_version).isdigit, rec_version)[:3])/100 #(get major,minor,fix of version)
    logger.debug("Exporting a video from recording with version: %s , %s"%(rec_version,rec_version_int))


    #load gaze information
    gaze_list = np.load(gaze_positions_path)
    timestamps = np.load(timestamps_path)
    #correlate data
    positions_by_frame = correlate_gaze(gaze_list,timestamps)


    # Initialize capture, check if it works
    cap = autoCreateCapture(video_path,timestamps=timestamps_path)
    if cap is None:
        logger.error("Did not receive valid Capture")
        return
    width,height = cap.get_size()

    #Out file path verification, we do this before but if one uses a seperate tool, this will kick in.
    if out_file_path is None:
        out_file_path = os.path.join(data_dir, "world_viz.avi")
    else:
        file_name =  os.path.basename(out_file_path)
        dir_name = os.path.dirname(out_file_path)
        if not dir_name:
            dir_name = data_dir
        if not file_name:
            file_name = 'world_viz.avi'
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

    #these two vars are shared with the lauching process and give an job lenght and progress report.
    frames_to_export.value = len(trimmed_timestamps)
    current_frame.value = 0
    logger.debug("Will export from frame %s to frame %s. This means I will export %s frames."%(start_frame,start_frame+frames_to_export.value,frames_to_export.value))


    #lets get the avg. framerate for our slice of video:
    fps = float(len(trimmed_timestamps))/(trimmed_timestamps[-1] - trimmed_timestamps[0])
    logger.debug("Framerate of export video is %s"%fps)


    #setup of writer
    writer = cv2.VideoWriter(out_file_path, cv2.cv.CV_FOURCC(*'DIVX'), fps, (width,height))

    cap.seek_to_frame(start_frame)

    start_time = time()


    plugins = []
    g = Temp()
    g.plugins = plugins
    g.app = 'exporter'

    # load plugins from initializers:
    for initializer in plugin_initializers:
        name, args = initializer
        logger.debug("Loading plugin: %s with settings %s"%(name, args))
        try:
            p = plugin_by_name[name](g,**args)
            plugins.append(p)
        except:
            logger.warning("Plugin '%s' could not be loaded in exporter." %name)


    while frames_to_export.value - current_frame.value > 0:

        if should_terminate.value:
            logger.warning("User aborted export. Exported %s frames to %s."%(current_frame.value,out_file_path))

            #explicit release of VideoWriter
            writer.release()
            writer = None
            return False

        new_frame = cap.get_frame()
        #end of video logic: pause at last frame.
        if not new_frame:
            logger.error("Could not read all frames.")
            #explicit release of VideoWriter
            writer.release()
            writer = None
            return False
        else:
            frame = new_frame

        #new positons and events
        current_pupil_positions = positions_by_frame[frame.index]
        events = None

        # allow each Plugin to do its work.
        for p in plugins:
            p.update(frame,current_pupil_positions,events)

        # # render gl visual feedback from loaded plugins
        # for p in plugins:
        #     p.gl_display(frame)

        writer.write(frame.img)
        current_frame.value +=1

    writer.release()
    writer = None

    duration = time()-start_time
    effective_fps = float(current_frame.value)/duration

    logger.info("Export done: Exported %s frames to %s. This took %s seconds. Exporter ran at %s frames per second"%(current_frame.value,out_file_path,duration,effective_fps))


    return True


