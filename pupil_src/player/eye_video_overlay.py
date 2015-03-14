'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys, os,platform
import cv2
import numpy as np
from file_methods import Persistent_Dict
from pyglui import ui
from pyglui.cygl.utils import create_named_texture,update_named_texture,draw_named_texture
from methods import normalize,denormalize
from plugin import Plugin
from glob import glob

from gl_utils import basic_gl_setup,adjust_gl_view, clear_gl_screen,make_coord_system_pixel_based,make_coord_system_norm_based

#capture
from uvc_capture import autoCreateCapture,EndofVideoFileError,FileSeekError,FakeCapture

#logging
import logging
logger = logging.getLogger(__name__)


def correlate_eye_world(eye_timestamps,world_timestamps):
    """
    args:
        eye_timestamps
        world_timestamps

    This function takes timestamps from eye and world processes as arguments
    and correlates the closest eye frame (or frames) by timestamp - similar to the `correlate_gaze` function in `player_methods`. 
    Returns: 
        `eye_frames_by_world_index` list - length of the list equals the number of frames in the world video.
    
    The eye process (even if captured at the same frame rate as the world video, e.g. 30Hz) typically will run slightly faster than the world process
    because of a smaller video capture and process load. In the future we will also have high speed eye cameras, therefore 
    there may be more than one valid timestamp for the eye video for each world frame, or no eye frame for a world frame. 

    Example:
    [[eye_timestamp, eye_timestamp, eye_timestamp],[eye_timestamp],[eye_timestamp,eye_timestamp],[],[eye_timestamp]...]


    This function gets called in the init of the plugin to create a lookup list called `eye_frame_by_world_index`. 
    
    The dictionary `eye_frames_by_timestamp` is also created in the plugin init
    with the `eye_timestamp` as the key and eye `frame_index` as value for convenient reverse lookup.
    """
    e_ts = eye_timestamps
    w_ts = list(world_timestamps)

    eye_frame_by_world_index = [[] for i in world_timestamps]

    frame_idx = 0
    try:
        current_e_ts = e_ts.pop(0)
    except:
        logger.warning("No eye timestamps found.")
        return eye_frame_by_world_index

    while e_ts:
        # if the current eye timestamp is before the mean of the current world frame timestamp and the next worldframe timestamp
        try:
            t_between_frames = ( w_ts[frame_idx]+w_ts[frame_idx+1] ) / 2.
        except IndexError:
            break
        if current_e_ts <= t_between_frames:
            eye_frame_by_world_index[frame_idx].append(current_e_ts)
            current_e_ts = e_ts.pop(0)
        else:
            frame_idx+=1

    return eye_frame_by_world_index


class Eye_Video_Overlay(Plugin):
    """docstring
    """
    def __init__(self,g_pool,menu_conf={}):
        super(Eye_Video_Overlay, self).__init__(g_pool)
        self.order = .2
        self.data_dir = g_pool.rec_dir
        self.menu_conf = menu_conf
        self.show_eye = False
        self._frame = None

        meta_info_path = self.data_dir + "/info.csv"

        #parse info.csv file
        with open(meta_info_path) as info:
            meta_info = dict( ((line.strip().split('\t')) for line in info.readlines() ) )
        rec_version = meta_info["Capture Software Version"]
        rec_version_float = int(filter(type(rec_version).isdigit, rec_version)[:3])/100. #(get major,minor,fix of version)
        eye_mode = meta_info["Eye Mode"]

        if rec_version_float < 0.4:
            required_files = ['eye.avi','eye_timestamps.npy']
            eye0_video_path = os.path.join(self.data_dir,required_files[0])
            eye0_timestamps_path = os.path.join(self.data_dir,required_files[1]) 
        else:
            required_files = ['eye0.mkv','eye0_timestamps.npy']
            eye0_video_path = os.path.join(self.data_dir,required_files[0])
            eye0_timestamps_path = os.path.join(self.data_dir,required_files[1])
            if eye_mode == 'binocular':
                required_files += ['eye1.mkv','eye1_timestamps.npy']
                eye1_video_path = os.path.join(self.data_dir,required_files[2])
                eye1_timestamps_path = os.path.join(self.data_dir,required_files[3])        

        # check to see if eye videos exist
        for f in required_files:
            if not os.path.isfile(os.path.join(self.data_dir,f)):
                logger.debug("Did not find required file: ") %(f, self.data_dir)
                self.cleanup() # early exit -- no required files

        logger.debug("%s contains required eye video(s): %s."%(self.data_dir,required_files))

        # Initialize capture -- for now we just try with monocular
        self.cap = autoCreateCapture(eye0_video_path,timestamps=eye0_timestamps_path)
       
        if isinstance(self.cap,FakeCapture):
            logger.error("could not start capture.")
            self.cleanup() # early exit -- no real eye videos

        self.width, self.height = self.cap.get_size()
        self._image_tex = create_named_texture((self.height,self.width,3))

        eye0_timestamps = list(np.load(eye0_timestamps_path))
        self.eye_frames_by_timestamp = dict(zip(eye0_timestamps,range(len(eye0_timestamps))))
        self.eye_frames_by_world_index = correlate_eye_world(eye0_timestamps,g_pool.timestamps)


    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Eye Video Overlay')
        # load the configuration of last session
        self.menu.configuration = self.menu_conf
        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self._update_gui()

    def unset_alive(self):
        self.alive = False

    def _update_gui(self):
        self.menu.elements[:] = []
        self.menu.append(ui.Info_Text('Show the eye video overlaid on top of the world video.'))
        self.menu.append(ui.Switch('show_eye',self,label='Show Eye Video'))        
        self.menu.append(ui.Button('close',self.unset_alive))

    def deinit_gui(self):
        if self.menu:
            self.menu_conf = self.menu.configuration
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        if self.menu:
            return {'menu_conf':self.menu.configuration}
        else:
            return {'menu_conf':self.menu_conf}

    def update(self,frame,events):
        # synchronize timestamps with world timestamps
        # frame.timestamp would be world frame timestamp

        # get 'pupil_positions' for the current timestamp - used to display pupil diameter
        # what is the last frame or should I seek next or repeat the frame
        # eye -- seek 
        # frame.idx (last frame index) - then skip seeking

        #grab new frame
        if self.g_pool.play or self.g_pool.new_seek:
            try:
                # new_frame = self.cap.get_frame()
                candidate_eye_frames = self.eye_frames_by_world_index[frame.index]
                if len(candidate_eye_frames) < 1:
                    # pop the last one off the list for the prior frame -- could be smarter 
                    candidate_eye_frames = self.eye_frames_by_world_index[frame.index-1][-1]
                else:
                    candidate_eye_frames = candidate_eye_frames[0]

                seek_pos = self.eye_frames_by_timestamp[candidate_eye_frames]
                print "seek_pos: ",seek_pos
                print "frame number: ",frame.index
                print "world time: ",frame.timestamp
                print "eye time: ",candidate_eye_frames

                # seek pos could be an empty list 
                self.cap.seek_to_frame(seek_pos)
                new_frame = self.cap.get_frame()
                
            except EndofVideoFileError:
                #end of video logic: pause at last frame.
                # g_pool.play=False
                print "reached the end of the eye video"

            self._frame = new_frame.copy()


    def gl_display(self):
        # update the eye texture 
        # render camera image
        if self._frame and self.show_eye:
            make_coord_system_norm_based()
            update_named_texture(self._image_tex,self._frame.img)
            draw_named_texture(self._image_tex,quad=((0.,0.),(.25,0.),(0.25,0.25),(0.,0.25)) )
            make_coord_system_pixel_based(self._frame.img.shape)
        # render visual feedback from loaded plugins

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()

        