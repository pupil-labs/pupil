'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

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

import cv2
import numpy as np
from uvc_capture import autoCreateCapture

#logging
import logging
logger = logging.getLogger(__name__)


def export(should_terminate,frames_to_export,current_frame, data_dir,start_frame=None,end_frame=None,plugins=None,out_file_path=None):

    #parse and load data dir info
    video_path = data_dir + "/world.avi"
    timestamps_path = data_dir + "/timestamps.npy"
    gaze_positions_path = data_dir + "/gaze_positions.npy"
    record_path = data_dir + "/world_viz.avi"


    #parse info.csv file
    with open(data_dir + "/info.csv") as info:
        meta_info = dict( ((line.strip().split('\t')) for line in info.readlines() ) )
    rec_version = meta_info["Capture Software Version"]
    rec_version_int = int(filter(type(rec_version).isdigit, rec_version)[:3]) #(get major,minor,fix of version)
    logger.debug("Recording version: %s , %s"%(rec_version,rec_version_int))


    #load gaze information
    gaze_list = list(np.load(gaze_positions_path))
    timestamps = list(np.load(timestamps_path))

    # this takes the timestamps list and makes a list
    # with the length of the number of recorded frames.
    # Each slot conains a list that will have 0, 1 or more assosiated gaze postions.
    positions_by_frame = [[] for i in timestamps]
    frame_idx = 0
    data_point = gaze_list.pop(0)
    gaze_timestamp = data_point[4]

    while gaze_list:
        # if the current gaze point is before the mean of the current world frame timestamp and the next worldframe timestamp
        try:
            t_between_frames = ( timestamps[frame_idx]+timestamps[frame_idx+1] ) / 2.
        except IndexError:
            break
        if gaze_timestamp <= t_between_frames:
            positions_by_frame[frame_idx].append({'norm_gaze':(data_point[0],data_point[1]),'norm_pupil': (data_point[2],data_point[3]), 'timestamp':gaze_timestamp})
            data_point = gaze_list.pop(0)
            gaze_timestamp = data_point[4]
        else:
            frame_idx+=1


    # Initialize capture, check if it works
    cap = autoCreateCapture(video_path,timestamps=timestamps_path)
    if cap is None:
        logger.error("Did not receive valid Capture")
        return
    width,height = cap.get_size()



    #Trim mark verification
    #make sure the trim marks (start frame, endframe) make sense: We define them like python list slices,thus we can test them like such.
    trimmed_timestamps = timestamps[start_frame:end_frame]
    if len(trimmed_timestamps)==0:
        logger.warn("Start and end frames are set such that no video will be exported.")
        return False

    if start_frame == None:
        start_frame = 0

    frames_to_export.value = len(trimmed_timestamps)
    current_frame.value = 0
    logger.debug("Will export from frame %s to frame %s. This means I will export %s frames."%(start_frame,start_frame+frames_to_export.value,frames_to_export.value))


    #lets get the avg. framerate for our slice of video:
    fps = float(len(trimmed_timestamps))/(trimmed_timestamps[-1] - trimmed_timestamps[0])
    logger.debug("Framerate of export video is %s"%fps)


    #Oout file path verification
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

    #setup of writer
    writer = cv2.VideoWriter(out_file_path, cv2.cv.CV_FOURCC(*'DIVX'), fps, (width,height))

    cap.seek_to_frame(start_frame)

    while frames_to_export.value - current_frame.value > 0 and not should_terminate.value:

        new_frame = cap.get_frame()
        #end of video logic: pause at last frame.
        if not new_frame:
            logger.error("Could not read all frames.")
            return False
        else:
            frame = new_frame

        #new positons and events
        current_pupil_positions = positions_by_frame[frame.index]
        events = None

        # allow each Plugin to do its work.
        for p in plugins:
            p.update(frame,current_pupil_positions,events)

        # render visual feedback from loaded plugins
        for p in plugins:
            p.img_display()

        writer.write(frame.img)
        current_frame.value +=1

    logger.debug("Export done: Exported %s frames to %s."%(current_frame.value,out_file_path))





if __name__ == '__main__':

    # make shared modules available across pupil_src
    from sys import path as syspath
    from os import path as ospath
    loc = ospath.abspath(__file__).rsplit('pupil_src', 1)
    syspath.append(ospath.join(loc[0], 'pupil_src', 'shared_modules'))
    del syspath, ospath


    from ctypes import  c_int,c_bool


    logging.basicConfig(level=logging.DEBUG)


    should_terminate = c_bool(False)
    frame_to_export  = c_int(0)
    current_frame = c_int(0)
    data_dir = '/Users/mkassner/Desktop/2014_01_21/000/'
    start_frame=200
    end_frame=300
    plugins=[]
    out_file_path="test.avi"


    export(should_terminate,frame_to_export,current_frame, data_dir,start_frame=start_frame,end_frame=end_frame,plugins=[],out_file_path=out_file_path)
    print current_frame.value

'''
exporter

    - is like a small player
    - launched with args:
        data folder
        start,end frame (trim marks)
        plugins loaded and their config
            - how to do this? 1) load the plugin instance as a whole?
                              2) create a plugin contructor based on a string or something similar?

    - can be used by batch or by player
    - communicates with progress (shared int) and terminate (shared bool)
    - can abort on demand leaving nothing behind
'''

