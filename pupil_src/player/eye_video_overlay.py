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
from player_methods import transparent_image_overlay
from plugin import Plugin

# helpers/utils
from version_utils import VersionFormat

#capture
from uvc_capture import autoCreateCapture,EndofVideoFileError,FileSeekError,FakeCapture

#logging
import logging
logger = logging.getLogger(__name__)


def get_past_timestamp(idx,timestamps):
    """
    recursive function to find the most recent valid timestamp in the past 
    """
    if idx == 0:
        # if at the beginning, we can't go back in time.
        return get_future_timestamp(idx,timestamps)
    if timestamps[idx]:
        res = timestamps[idx][-1]
        return res
    else:
        return get_past_timestamp(idx-1,timestamps)

def get_future_timestamp(idx,timestamps):
    """
    recursive function to find most recent valid timestamp in the future
    """    
    if idx == len(timestamps)-1:
        # if at the end, we can't go further into the future.
        return get_past_timestamp(idx,timestamps)
    elif timestamps[idx]:
        return timestamps[idx][0]
    else:
        idx = min(len(timestamps),idx+1)
        return get_future_timestamp(idx,timestamps)

def get_nearest_timestamp(past_timestamp,future_timestamp,world_timestamp):
    dt_past = abs(past_timestamp-world_timestamp)
    dt_future = abs(future_timestamp-world_timestamp) # abs prob not necessary here, but just for sanity 
    if dt_past < dt_future:
        return past_timestamp
    else: 
        return future_timestamp 


def correlate_eye_world(eye_timestamps,world_timestamps):
    """
    This function takes a list of eye timestamps and world timestamps
    and correlates one eye frame per world frame
    Returns a list of eye indicies `eye_frame_index` that has length equal the world frame index
    """
    # return framewise mapping as a list
    e_ts = eye_timestamps
    w_ts = list(world_timestamps)
    eye_frames_by_timestamp = dict(zip(e_ts,range(len(e_ts))))

    eye_timestamps_by_world_index = [[] for i in world_timestamps]

    frame_idx = 0
    try:
        current_e_ts = e_ts.pop(0)
    except:
        logger.warning("No eye timestamps found.")
        return eye_timestamps_by_world_index

    while e_ts:
        # if the current eye timestamp is before the mean of the current world frame timestamp and the next worldframe timestamp
        try:
            t_between_frames = ( w_ts[frame_idx]+w_ts[frame_idx+1] ) / 2.
        except IndexError:
            break
        if current_e_ts <= t_between_frames:
            eye_timestamps_by_world_index[frame_idx].append(current_e_ts)
            current_e_ts = e_ts.pop(0)
        else:
            frame_idx+=1

    idx = 0
    eye_frame_index = []
    # some entiries in the `eye_timestamps_by_world_index` might be empty -- no correlated eye timestamp
    # so we will either show the previous frame or next frame - whichever is temporally closest
    for candidate,world_ts in zip(eye_timestamps_by_world_index,w_ts):
        # if there is no candidate, then assign it to the closest timestamp
        if not candidate:
            # get most recent timestamp, either in the past or future
            e_past_ts = get_past_timestamp(idx,eye_timestamps_by_world_index)
            e_future_ts = get_future_timestamp(idx,eye_timestamps_by_world_index)        
            eye_frame_index.append(eye_frames_by_timestamp[get_nearest_timestamp(e_past_ts,e_future_ts,world_ts)]) 
        else:
            # TODO - if there is a list of len > 1 - then we should check which is the temporally closest timestamp
            eye_frame_index.append(eye_frames_by_timestamp[eye_timestamps_by_world_index[idx][-1]])

        idx += 1

    return eye_frame_index


class Eye_Video_Overlay(Plugin):
    """docstring
    """
    def __init__(self,g_pool,alpha=0.6,mirror=True,menu_conf={'collapsed':False}):
        super(Eye_Video_Overlay, self).__init__(g_pool)
        self.order = .6

        # initialize empty menu
        # and load menu configuration of last session
        self.menu = None
        self.menu_conf = menu_conf
        
        # user controls
        self.alpha = alpha
        self.mirror = mirror

        self.last_world_idx = None 
        self._frame = None

        #parse info.csv file
        with open(g_pool.rec_dir + "/info.csv") as info:
            meta_info = dict( ((line.strip().split('\t')) for line in info.readlines() ) )

        # load eye videos and eye timestamps 
        if g_pool.rec_version < VersionFormat('0.4'):
            required_files = ['eye.avi','eye_timestamps.npy']
            eye0_video_path = os.path.join(g_pool.rec_dir,required_files[0])
            eye0_timestamps_path = os.path.join(g_pool.rec_dir,required_files[1]) 
        else:
            required_files = ['eye0.mkv','eye0_timestamps.npy']
            eye0_video_path = os.path.join(g_pool.rec_dir,required_files[0])
            eye0_timestamps_path = os.path.join(g_pool.rec_dir,required_files[1])
            if meta_info['Eye Mode'] == 'binocular':
                required_files += ['eye1.mkv','eye1_timestamps.npy']
                eye1_video_path = os.path.join(g_pool.rec_dir,required_files[2])
                eye1_timestamps_path = os.path.join(g_pool.rec_dir,required_files[3])        

        # check to see if eye videos exist
        for f in required_files:
            if not os.path.isfile(os.path.join(g_pool.rec_dir,f)):
                logger.debug("Did not find required file: ") %(f, g_pool.rec_dir)
                self.cleanup() # early exit -- no required files

        logger.debug("%s contains required eye video(s): %s."%(g_pool.rec_dir,required_files))

        # Initialize capture -- for now we just try with monocular
        self.cap = autoCreateCapture(eye0_video_path,timestamps=eye0_timestamps_path)
       
        if isinstance(self.cap,FakeCapture):
            logger.error("could not start capture.")
            self.cleanup() # early exit -- no real eye videos

        self.width, self.height = self.cap.get_size()
        self._image_tex = create_named_texture((self.height,self.width,3))

        eye0_timestamps = list(np.load(eye0_timestamps_path))
        self.eye0_frame_index = correlate_eye_world(eye0_timestamps,g_pool.timestamps)



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
        self.menu.append(ui.Slider('alpha',self,min=0.0,step=0.05,max=1.0,label='Opacity'))
        self.menu.append(ui.Switch('mirror',self,label="Mirror image"))
        self.menu.append(ui.Button('close',self.unset_alive))

    def deinit_gui(self):
        if self.menu:
            self.menu_conf = self.menu.configuration
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def update(self,frame,events):
        requested_eye_frame_idx = self.eye0_frame_index[frame.index]
        
        if self.last_world_idx != frame.index and requested_eye_frame_idx != frame.index:
            # seeklogic
            if requested_eye_frame_idx == self.cap.get_frame_index()+1:
                # if we just need to seek by one frame, its faster to just read one and and throw it away.
                _ = self.cap.get_frame()
            if requested_eye_frame_idx != self.cap.get_frame_index():
               # only now do I need to seek
               self.cap.seek_to_frame(requested_eye_frame_idx)
            
            # reading the frame
            try:
               self._frame = self.cap.get_frame()
            except EndofVideoFileError:
                logger.warning("Reached the end of the eye video.")
            self.last_world_idx = frame.index
        else:
            #our old frame is still valid because we are doing upsampling
            pass

        pad = 10
        pos = frame.width-self.width-pad, pad

        if self._frame is not None:
            transparent_image_overlay(pos,np.fliplr(self._frame.img) if self.mirror else self._frame.img,frame.img,self.alpha)

    def gl_display(self):
        # removed texture method because we need to be able to see what we will export - draw directly in the array
        # update the eye texture 
        # render camera image
        # if self._frame and self.show_eye:
        #     print self._frame.img.shape
        #     make_coord_system_norm_based()
        #     update_named_texture(self._image_tex,self._frame.img)
        #     draw_named_texture(self._image_tex,quad=((0.,0.),(.25,0.),(0.25,0.25),(0.,0.25)) )
        #     make_coord_system_pixel_based(self._frame.img.shape)
        # render visual feedback from loaded plugins
        pass


    def get_init_dict(self):
        if self.menu:
            return {'alpha':self.alpha,'mirror':self.mirror,'menu_conf':self.menu.configuration}
        else:
            return {'alpha':self.alpha,'mirror':self.mirror,'menu_conf':self.menu_conf}

    def clone(self):
        return Eye_Video_Overlay(**self.get_init_dict())

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()

        
