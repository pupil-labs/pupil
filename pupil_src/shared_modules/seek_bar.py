'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from pyglui.cygl.utils import draw_polyline,draw_points,RGBA

from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D

from glfw import glfwGetWindowSize,glfwGetCurrentContext,glfwGetCursorPos,GLFW_RELEASE,GLFW_PRESS,glfwGetFramebufferSize
from plugin import Plugin

import logging
logger = logging.getLogger(__name__)

class Seek_Bar(Plugin):
    """docstring for Seek_Bar
    seek bar displays a bar at the bottom of the screen when you hover close to it.
    it will show the current positon and allow you to drag to any postion in the video file.
    """
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.cap = g_pool.capture
        self.current_frame_index = self.cap.get_frame_index()
        self.frame_count = self.cap.get_frame_count()

        self.drag_mode = False
        self.was_playing = True
        #display layout
        self.padding = 30. #in sceen pixel
        self.window_size = 0,0


    def init_gui(self):
        self.on_window_resize(glfwGetCurrentContext(),*glfwGetWindowSize(glfwGetCurrentContext()))

    def on_window_resize(self,window,w,h):
        self.window_size = w,h
        self.h_pad = self.padding * self.frame_count/float(w)
        self.v_pad = self.padding * 1./h

    def recent_events(self,events):
        frame = events.get('frame')
        if not frame:
            return

        self.current_frame_index = frame.index

        if self.drag_mode:
            x, y = glfwGetCursorPos(glfwGetCurrentContext())
            x, _ = self.screen_to_seek_bar((x,y))
            seek_pos = min(self.frame_count,max(0,x))
            seek_pos = int(min(seek_pos,self.frame_count-5)) #the last frames can be problematic to seek to
            if self.current_frame_index-1 != seek_pos:
                try:
                    # logger.info('seeking to {} form {}'.format(seek_pos,self.current_frame_index))
                    self.cap.seek_to_frame(seek_pos)
                    self.current_frame_index = self.cap.get_frame_index() + 1
                except:
                    pass
            self.g_pool.new_seek = True

    def on_click(self,img_pos,button,action):
        """
        gets called when the user clicks in the window screen
        """
        hdpi_factor = float(glfwGetFramebufferSize(glfwGetCurrentContext())[0]/glfwGetWindowSize(glfwGetCurrentContext())[0])
        pos = glfwGetCursorPos(glfwGetCurrentContext())
        pos = pos[0]*hdpi_factor,pos[1]*hdpi_factor
        #drag the seek point
        if action == GLFW_PRESS:
            screen_seek_pos = self.seek_bar_to_screen((self.current_frame_index,0))
            dist = abs(pos[0]-screen_seek_pos[0])+abs(pos[1]-screen_seek_pos[1])
            if dist < 20:
                self.drag_mode=True
                self.was_playing = self.g_pool.play
                self.g_pool.play = False

        elif action == GLFW_RELEASE:
            if self.drag_mode:
                x, _ = self.screen_to_seek_bar(pos)
                x = int(min(self.frame_count-5,max(0,x)))
                try:
                    self.cap.seek_to_frame(x)
                except:
                    pass
                self.g_pool.new_seek = True
                self.drag_mode=False
                self.g_pool.play = self.was_playing

    def seek_bar_to_screen(self,pos):
        width,height = self.window_size
        x,y=pos
        y = 1-y
        x = (x/float(self.frame_count))*(width-self.padding*2) +self.padding
        y  = y*(height-2*self.padding)+self.padding
        return x,y

    def screen_to_seek_bar(self,pos):
        width,height = glfwGetWindowSize(glfwGetCurrentContext())
        x,y=pos
        x  = (x-self.padding)/(width-2*self.padding)*self.frame_count
        y  = (y-self.padding)/(height-2*self.padding)
        return x,1-y

    def gl_display(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(-self.h_pad,  (self.frame_count)+self.h_pad, -self.v_pad, 1+self.v_pad) # ranging from 0 to cache_len-1 (horizontal) and 0 to 1 (vertical)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # if self.drag_mode:
        #     color1 = (0.,.8,.5,.5)
        #     color2 = (0.,.8,.5,1.)
        # else:
        color1 = (1,1,1,0.4)#(.25,.8,.8,.5)
        color2 = (1,1,1,1.)#(.25,.8,.8,1.)

        thickness = 10.
        draw_polyline(verts=[(0,0),(self.current_frame_index,0)],
            thickness=thickness,color=RGBA(*color1))
        draw_polyline(verts=[(self.current_frame_index,0),(self.frame_count,0)],
            thickness=thickness,color=RGBA(*color1))
        if not self.drag_mode:
            draw_points([(self.current_frame_index,0)],color=RGBA(*color1),size=30)
        draw_points([(self.current_frame_index,0)],color=RGBA(*color2),size=20)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
