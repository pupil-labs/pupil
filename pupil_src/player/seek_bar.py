'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from gl_utils import draw_gl_polyline,draw_gl_point
from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D

from glfw import glfwGetWindowSize,glfwGetCurrentContext,glfwGetCursorPos,GLFW_RELEASE,GLFW_PRESS
from plugin import Plugin
import numpy as np

from methods import denormalize
import logging
logger = logging.getLogger(__name__)

class Seek_Bar(Plugin):
    """docstring for Seek_Bar
    seek bar displays a bar at the bottom of the screen when you hover close to it.
    it will show the current positon and allow you to drag to any postion in the video file.

    """
    def __init__(self, g_pool,capture):
        super(Seek_Bar, self).__init__()
        self.g_pool = g_pool
        self.cap = capture
        self.current_frame_index = self.cap.get_frame_index()
        self.frame_count = self.cap.get_frame_count()

        self.norm_seek_pos = self.current_frame_index/float(self.frame_count)
        self.drag_mode = False
        self.was_playing = True
        #display layout
        self.padding = 20.

    def update(self,frame,recent_pupil_positions,events):
        self.current_frame_index = frame.index
        self.norm_seek_pos = self.current_frame_index/float(self.frame_count)

        if self.drag_mode:
            pos = glfwGetCursorPos(glfwGetCurrentContext())
            norm_seek_pos, _ = self.screen_to_seek_bar(pos)
            norm_seek_pos = min(1,max(0,norm_seek_pos))
            if abs(norm_seek_pos-self.norm_seek_pos) >=.01:
                seek_pos = int(norm_seek_pos*self.frame_count)
                self.cap.seek_to_frame(seek_pos)
                self.g_pool.new_seek = True



    def on_click(self,img_pos,button,action):
        """
        gets called when the user clicks in the window screen
        """
        pos = glfwGetCursorPos(glfwGetCurrentContext())
        #drag the seek point
        if action == GLFW_PRESS:
            screen_seek_pos = self.seek_bar_to_screen((self.norm_seek_pos,0))
            dist = abs(pos[0]-screen_seek_pos[0])+abs(pos[1]-screen_seek_pos[1])
            if dist < 20:
                self.drag_mode=True
                self.was_playing = self.g_pool.play
                self.g_pool.play = False

        elif action == GLFW_RELEASE:
            if self.drag_mode:
                self.drag_mode=False
                self.g_pool.play = self.was_playing


    def seek_bar_to_screen(self,pos):
        width,height = glfwGetWindowSize(glfwGetCurrentContext())
        x,y=pos
        y = 1-y
        x  = x*(width-2*self.padding)+self.padding
        y  = y*(height-2*self.padding)+self.padding
        return x,y


    def screen_to_seek_bar(self,pos):
        width,height = glfwGetWindowSize(glfwGetCurrentContext())
        x,y=pos
        x  = (x-self.padding)/(width-2*self.padding)
        y  = (y-self.padding)/(height-2*self.padding)
        return x,1-y

    def gl_display(self):

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        width,height = glfwGetWindowSize(glfwGetCurrentContext())
        h_pad = self.padding/width
        v_pad = self.padding/height
        gluOrtho2D(-h_pad, 1+h_pad, -v_pad, 1+v_pad) # gl coord convention
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        draw_gl_point((self.norm_seek_pos,0))
        draw_gl_polyline( [(0,0),(1,0)],color=(1,0,0,1))

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()


