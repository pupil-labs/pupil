'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import os
import numpy as np

#opengl
from OpenGL.GL import *
from glfw import glfwGetWindowSize,glfwGetCurrentContext,glfwGetCursorPos,GLFW_RELEASE,GLFW_PRESS,glfwGetFramebufferSize

#logging
import logging
logger = logging.getLogger(__name__)

from pyglui import ui
from pyglui.cygl.utils import draw_polyline,draw_points,RGBA
from plugin import Plugin
from file_methods import load_object
from player_methods import correlate_data

class Event_Player(Plugin):
    def __init__(self,g_pool):
        super(Event_Player, self).__init__(g_pool)

        self.menu = None
        self.frame_count = g_pool.capture.get_frame_index()

        #display layout
        self.padding = 20. #in sceen pixel
        self.window_size = 0,0


    def get_index(self):
        captured_events = load_object(os.path.join(self.g_pool.rec_dir, "user_addition_events"))
        dict_captured_events = [{'timestamp':i[1]} for i in captured_events]
        np_timestamps = np.load(os.path.join(self.g_pool.rec_dir, "world_timestamps.npy"))
        timestamps = np_timestamps.tolist()
        
        data = correlate_data(dict_captured_events, timestamps)

        return data


    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Event Player')

        # add menu to the window
        self.g_pool.gui.append(self.menu)

        self.menu.append(ui.Button('remove',self.unset_alive))

        self.on_window_resize(glfwGetCurrentContext(),*glfwGetWindowSize(glfwGetCurrentContext()))


    def on_window_resize(self,window,w,h):
        self.window_size = w,h
        self.h_pad = self.padding * self.frame_count/float(w)
        self.v_pad = self.padding * 1./h


    def update(self,frame,events):
        pass


    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None


    def unset_alive(self):
        self.alive = False


    def gl_display(self):

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(-self.h_pad, (self.frame_count)+self.h_pad, -self.v_pad, 1+self.v_pad,-1,1) # ranging from 0 to cache_len-1 (horizontal) and 0 to 1 (vertical)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        event_tick_color = (1.,0.,0.,.8)
        events_in_frame = self.get_index()

        for frame in events_in_frame:
            logger.info(frame)
            if len(frame) != 0:
                draw_polyline(verts=[(frame[0]['index'],0),(frame[0]['index'],20)],
                    color=RGBA(*event_tick_color))


        # draw_polyline(verts=[(0,0),(self.current_frame_index,0)],color=RGBA(*color1))
        # draw_polyline(verts=[(self.current_frame_index,0),(self.frame_count,0)],color=RGBA(.5,.5,.5,.5))
        # draw_points([(self.current_frame_index,0)],color=RGBA(*color1),size=40)
        # draw_points([(self.current_frame_index,0)],color=RGBA(*color2),size=10)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()


    def cleanup(self):
        """called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
