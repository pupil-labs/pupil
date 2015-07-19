'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from OpenGL.GL import *
from pyglui.cygl.utils import RGBA,draw_points,draw_polyline
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_roboto_font_path
from glfw import glfwGetWindowSize,glfwGetCurrentContext,glfwGetCursorPos,GLFW_RELEASE,GLFW_PRESS,glfwGetFramebufferSize
from plugin import Plugin

import logging
logger = logging.getLogger(__name__)

class Trim_Marks(Plugin):
    """docstring for Trim_Mark
    """
    def __init__(self, g_pool):
        super(Trim_Marks, self).__init__(g_pool)
        g_pool.trim_marks = self #attach self for ease of acces by others.
        self.order = .8
        self.capture = g_pool.capture
        self.frame_count = self.capture.get_frame_count()
        self._in_mark = 0
        self._out_mark = self.frame_count
        self.drag_in = False
        self.drag_out = False
        #display layout
        self.padding = 20. #in screen pixel
        self.window_size = 0,0

        self.frame_size = self.capture.frame_size
        # on drag text
        self.text = ['','']
        self.glfont = fontstash.Context()
        self.glfont.add_font('roboto',get_roboto_font_path())
        self.glfont.set_size(self.frame_size[0]/30)
        self.glfont.set_color_float((0.7,0.7,0.7,1.0))

    @property
    def in_mark(self):
        return self._in_mark

    @in_mark.setter
    def in_mark(self, value):
        self._in_mark = int(min(self._out_mark,max(0,value)))

    @property
    def out_mark(self):
        return self._out_mark

    @out_mark.setter
    def out_mark(self, value):
        self._out_mark = int(min(self.frame_count,max(self.in_mark,value)))

    def set(self,mark_range):
        self._in_mark,self._out_mark = mark_range

    def get_string(self):
        return '%s - %s'%(self._in_mark,self._out_mark)

    def set_string(self,str):
        try:
            in_m,out_m = str.split('-')
            in_m = int(in_m)
            out_m = int(out_m)
            self.in_mark = in_m
            self.out_mark = out_m
        except:
            logger.warning("Setting Trimmarks via string failed.")
    def init_gui(self):
        self.on_window_resize(glfwGetCurrentContext(),*glfwGetWindowSize(glfwGetCurrentContext()))

    def on_window_resize(self,window,w,h):
        self.window_size = w,h
        self.h_pad = self.padding * self.frame_count/float(w)
        self.v_pad = self.padding * 1./h

    def update(self,frame,events):
        if frame.index == self.out_mark or frame.index == self.in_mark:
            self.g_pool.play=False

        if self.drag_in:
            x,y = glfwGetCursorPos(glfwGetCurrentContext())
            x,_ = self.screen_to_bar_space((x,y))
            self.in_mark = x
            self.text[0] = str(self.in_mark)
            
        elif self.drag_out:
            x,y = glfwGetCursorPos(glfwGetCurrentContext())
            x,_ = self.screen_to_bar_space((x,y))
            self.out_mark = x
            self.text[1] = str(self.out_mark)

    def on_click(self,img_pos,button,action):
        """
        gets called when the user clicks in the window screen
        """
        hdpi_factor = float(glfwGetFramebufferSize(glfwGetCurrentContext())[0]/glfwGetWindowSize(glfwGetCurrentContext())[0])
        pos = glfwGetCursorPos(glfwGetCurrentContext())
        pos = pos[0]*hdpi_factor,pos[1]*hdpi_factor

        #drag the seek point
        if action == GLFW_PRESS:
            screen_in_mark_pos = self.bar_space_to_screen((self.in_mark,0))
            screen_out_mark_pos = self.bar_space_to_screen((self.out_mark,0))

            #in mark
            dist = abs(pos[0]-screen_in_mark_pos[0])+abs(pos[1]-screen_in_mark_pos[1])
            if dist < 10:
                if self.distance_in_pix(self.in_mark,self.capture.get_frame_index()) > 20:
                    self.drag_in=True
                    return
            #out mark
            dist = abs(pos[0]-screen_out_mark_pos[0])+abs(pos[1]-screen_out_mark_pos[1])
            if dist < 10:
                if self.distance_in_pix(self.out_mark,self.capture.get_frame_index()) > 20:
                    self.drag_out=True

        elif action == GLFW_RELEASE:
            self.drag_out = False
            self.drag_in = False


    def distance_in_pix(self,frame_pos_0,frame_pos_1):
        fr0_screen_x,_ = self.bar_space_to_screen((frame_pos_0,0))
        fr1_screen_x,_ = self.bar_space_to_screen((frame_pos_1,0))
        return abs(fr0_screen_x-fr1_screen_x)

    def bar_space_to_screen(self,pos):
        width,height = self.window_size
        x,y=pos
        y = 1-y
        x = (x/float(self.frame_count))*(width-self.padding*2) +self.padding
        y  = y*(height-2*self.padding)+self.padding
        return x,y

    def screen_to_bar_space(self,pos):
        width,height = glfwGetWindowSize(glfwGetCurrentContext())
        x,y=pos
        x  = (x-self.padding)/(width-2*self.padding)*self.frame_count
        y  = (y-self.padding)/(height-2*self.padding)
        return x,1-y

    def gl_display(self):
        # still need appropriate padding and resizing
        w, h = self.frame_size
        if self.drag_in:
            self.glfont.draw_text((w/8),h-(h/20),self.text[0])
        elif self.drag_out:
            self.glfont.draw_text((w/8)*7,h-(h/20),self.text[1])

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()

        glOrtho(-self.h_pad,  (self.frame_count)+self.h_pad, -self.v_pad, 1+self.v_pad,-1,1) # ranging from 0 to cache_len-1 (horizontal) and 0 to 1 (vertical)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        color1 = RGBA(.1,.9,.2,.5)
        color2 = RGBA(.1,.9,.2,.5)

        if self.in_mark != 0 or self.out_mark != self.frame_count:
            draw_polyline( [(self.in_mark,0),(self.out_mark,0)],color=color1,thickness=2)
        draw_points([(self.in_mark,0),],color=color2,size=10)
        draw_points([(self.out_mark,0),],color=color2,size=10)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
