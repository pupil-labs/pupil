'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import sys, os, platform
from glob import glob
import cv2
import math
import numpy as np
from file_methods import Persistent_Dict
from pyglui import ui
from player_methods import transparent_image_overlay
from plugin import Visualizer_Plugin_Base
from copy import copy

# helpers/utils
from version_utils import VersionFormat

#capture
from video_capture import EndofVideoFileError,FileSeekError,FileCaptureError,File_Source

#mouse
from glfw import glfwGetCursorPos,glfwGetWindowSize,glfwGetCurrentContext
from methods import normalize,denormalize

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
    Returns a mapping that correlates a single eye frame index with each world frame index.
    Up and downsampling is used to achieve this mapping.
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
    eye_world_frame_map = []
    # some entiries in the `eye_timestamps_by_world_index` might be empty -- no correlated eye timestamp
    # so we will either show the previous frame or next frame - whichever is temporally closest
    for candidate,world_ts in zip(eye_timestamps_by_world_index,w_ts):
        # if there is no candidate, then assign it to the closest timestamp
        if not candidate:
            # get most recent timestamp, either in the past or future
            e_past_ts = get_past_timestamp(idx,eye_timestamps_by_world_index)
            e_future_ts = get_future_timestamp(idx,eye_timestamps_by_world_index)
            eye_world_frame_map.append(eye_frames_by_timestamp[get_nearest_timestamp(e_past_ts,e_future_ts,world_ts)])
        else:
            # TODO - if there is a list of len > 1 - then we should check which is the temporally closest timestamp
            eye_world_frame_map.append(eye_frames_by_timestamp[eye_timestamps_by_world_index[idx][-1]])
        idx += 1

    return eye_world_frame_map

class Vis_Eye_Video_Overlay(Visualizer_Plugin_Base):
    """docstring This plugin allows the user to overlay the eye recording on the recording of his field of vision
        Features: flip video across horiz/vert axes, click and drag around interface, scale video size from 20% to 100%,
        show only 1 or 2 or both eyes
        features updated by Andrew June 2015
    """
    def __init__(self,g_pool,alpha=0.6,eye_scale_factor=.5,move_around=0,mirror={'0':False,'1':False}, flip={'0':False,'1':False},pos=[(640,10),(10,10)], show_ellipses=True):
        super().__init__(g_pool)
        self.order = .6
        self.menu = None

        # user controls
        self.alpha = alpha #opacity level of eyes
        self.eye_scale_factor = eye_scale_factor #scale
        self.showeyes = 0,1 #modes: any text containg both means both eye is present, on 'only eye1' if only one eye recording
        self.move_around = move_around #boolean whether allow to move clip around screen or not
        self.video_size = [0,0] #video_size of recording (bc scaling)
        self.show_ellipses = show_ellipses

        #variables specific to each eye
        self.eye_frames = []
        self.eye_world_frame_map = []
        self.eye_cap = []
        self.mirror = mirror #do we horiz flip first eye
        self.flip = flip #do we vert flip first eye
        self.pos = [list(pos[0]),list(pos[1])] #positions of 2 eyes
        self.drag_offset = [None,None]

        # load eye videos and eye timestamps
        if VersionFormat(self.g_pool.meta_info['Capture Software Version'][1:]) < VersionFormat('0.4'):
            eye_video_path = os.path.join(g_pool.rec_dir,'eye.avi'),'None'
            eye_timestamps_path = os.path.join(g_pool.rec_dir,'eye_timestamps.npy'),'None'
        else:
            eye_video_path = os.path.join(g_pool.rec_dir,'eye0.*'),os.path.join(g_pool.rec_dir,'eye1.*')
            eye_timestamps_path = os.path.join(g_pool.rec_dir,'eye0_timestamps.npy'),os.path.join(g_pool.rec_dir,'eye1_timestamps.npy')

        #try to load eye video and ts for each eye.
        for video,ts in zip(eye_video_path,eye_timestamps_path):
            try:
                class empty(object):
                    pass
                self.eye_cap.append(File_Source(empty(),source_path=glob(video)[0],timestamps=np.load(ts)))
            except(IndexError,FileCaptureError):
                pass
            else:
                self.eye_frames.append(self.eye_cap[-1].get_frame())
            try:
                eye_timestamps = list(np.load(ts))
            except:
                pass
            else:
                self.eye_world_frame_map.append(correlate_eye_world(eye_timestamps,g_pool.timestamps))

        if len(self.eye_cap) == 2:
            logger.debug("Loaded binocular eye video data.")
        elif len(self.eye_cap) == 1:
            logger.debug("Loaded monocular eye video data")
            self.showeyes = (0,)
        else:
            logger.error("Could not load eye video.")
            self.alive = False
            return

    def unset_alive(self):
        self.alive = False

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Eye Video Overlay')
        self.update_gui()
        self.g_pool.gui.append(self.menu)

    def update_gui(self):
        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',self.unset_alive))
        self.menu.append(ui.Info_Text('Show the eye video overlaid on top of the world video. Eye1 is usually the right eye'))
        self.menu.append(ui.Slider('alpha',self,min=0.0,step=0.05,max=1.0,label='Opacity'))
        self.menu.append(ui.Slider('eye_scale_factor',self,min=0.2,step=0.1,max=1.0,label='Video Scale'))
        self.menu.append(ui.Switch('move_around',self,label="Move Overlay"))
        if len(self.eye_cap) == 2:
            self.menu.append(ui.Selector('showeyes',self,label='Show',selection=[(0,),(1,),(0,1)],labels= ['Eye 0','Eye 1','both'],setter=self.set_showeyes))
        if 0 in self.showeyes:
            self.menu.append(ui.Switch('0',self.mirror,label="Eye 0: Horiz. Flip"))
            self.menu.append(ui.Switch('0',self.flip,label="Eye 0: Vert. Flip"))
        if 1 in self.showeyes:
            self.menu.append(ui.Switch('1',self.mirror,label="Eye 1: Horiz Flip"))
            self.menu.append(ui.Switch('1',self.flip,label="Eye 1: Vert Flip"))
        self.menu.append(ui.Switch('show_ellipses', self, label="Visualize Ellipses"))


    def set_showeyes(self,new_mode):
        #everytime we choose eye setting (either use eye 1, 2, or both, updates the gui menu to remove certain options from list)
        self.showeyes = new_mode
        self.update_gui()

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        for eye_index in self.showeyes:
            requested_eye_frame_idx = self.eye_world_frame_map[eye_index][frame.index]

            #1. do we need a new frame?
            if requested_eye_frame_idx != self.eye_frames[eye_index].index:
                # do we need to seek?
                if requested_eye_frame_idx == self.eye_cap[eye_index].get_frame_index()+1:
                    # if we just need to seek by one frame, its faster to just read one and and throw it away.
                    _ = self.eye_cap[eye_index].get_frame()
                if requested_eye_frame_idx != self.eye_cap[eye_index].get_frame_index():
                    # only now do I need to seek
                    self.eye_cap[eye_index].seek_to_frame(requested_eye_frame_idx)
                # reading the new eye frame frame
                try:
                    self.eye_frames[eye_index] = self.eye_cap[eye_index].get_frame()
                except EndofVideoFileError:
                    logger.warning("Reached the end of the eye video for eye video {}.".format(eye_index))
            else:
                #our old frame is still valid because we are doing upsampling
                pass

            #2. dragging image
            if self.drag_offset[eye_index] is not None:
                pos = glfwGetCursorPos(glfwGetCurrentContext())
                pos = normalize(pos,glfwGetWindowSize(glfwGetCurrentContext()))
                pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels
                self.pos[eye_index][0] = pos[0]+self.drag_offset[eye_index][0]
                self.pos[eye_index][1] = pos[1]+self.drag_offset[eye_index][1]
            else:
                self.video_size = [round(self.eye_frames[eye_index].width*self.eye_scale_factor), round(self.eye_frames[eye_index].height*self.eye_scale_factor)]

            #3. keep in image bounds, do this even when not dragging because the image video_sizes could change.
            self.pos[eye_index][1] = min(frame.img.shape[0]-self.video_size[1],max(self.pos[eye_index][1],0)) #frame.img.shape[0] is height, frame.img.shape[1] is width of screen
            self.pos[eye_index][0] = min(frame.img.shape[1]-self.video_size[0],max(self.pos[eye_index][0],0))

            #4. flipping images, converting to greyscale
            eye_gray = cv2.cvtColor(self.eye_frames[eye_index].img,cv2.COLOR_BGR2GRAY) #auto gray scaling
            eyeimage = cv2.resize(eye_gray,(0,0),fx=self.eye_scale_factor, fy=self.eye_scale_factor)
            if self.mirror[str(eye_index)]:
                eyeimage = np.fliplr(eyeimage)
            if self.flip[str(eye_index)]:
                eyeimage = np.flipud(eyeimage)

            eyeimage = cv2.cvtColor(eyeimage, cv2.COLOR_GRAY2BGR)

            if self.show_ellipses and events['pupil_positions']:
                for pd in events['pupil_positions']:
                    if pd['id'] == eye_index and pd['timestamp'] == self.eye_frames[eye_index].timestamp:
                        el = pd['ellipse']
                        conf = int(pd.get('model_confidence', pd.get('confidence', 0.1)) * 255)
                        center = list(map(lambda val: int(self.eye_scale_factor*val), el['center']))
                        el['axes'] = tuple(map(lambda val: int(self.eye_scale_factor*val/2), el['axes']))
                        el['angle'] = int(el['angle'])
                        el_points = cv2.ellipse2Poly(tuple(center), el['axes'], el['angle'], 0, 360, 1)

                        if self.mirror[str(eye_index)]:
                            el_points = [(self.video_size[0] - x, y) for x, y in el_points]
                            center[0] = self.video_size[0] - center[0]
                        if self.flip[str(eye_index)]:
                            el_points = [(x, self.video_size[1] - y) for x, y in el_points]
                            center[1] = self.video_size[1] - center[1]

                        cv2.polylines(eyeimage, [np.asarray(el_points)], True, (0, 0, 255, conf), thickness=math.ceil(2*self.eye_scale_factor))
                        cv2.circle(eyeimage, tuple(center), int(5*self.eye_scale_factor), (0, 0, 255, conf), thickness=-1)

            # 5. finally overlay the image
            x, y = int(self.pos[eye_index][0]), int(self.pos[eye_index][1])
            transparent_image_overlay((x, y), eyeimage, frame.img, self.alpha)

    def on_click(self,pos,button,action):
        if self.move_around == 1 and action == 1:
            for eye_index in self.showeyes:
                if self.pos[eye_index][0] < pos[0] < self.pos[eye_index][0]+self.video_size[0] and self.pos[eye_index][1] < pos[1] < self.pos[eye_index][1] + self.video_size[1]:
                    self.drag_offset[eye_index] = self.pos[eye_index][0]-pos[0],self.pos[eye_index][1]-pos[1]
                    return
        else:
            self.drag_offset = [None,None]

    def get_init_dict(self):
        return {'alpha':self.alpha,'eye_scale_factor':self.eye_scale_factor,'move_around':self.move_around,'mirror':self.mirror,'flip':self.flip,'pos':self.pos,'move_around':self.move_around, 'show_ellipses': self.show_ellipses}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
