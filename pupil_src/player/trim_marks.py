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
from glfw import glfwGetWindowSize,glfwGetCurrentContext,glfwGetCursorPos,GLFW_RELEASE,GLFW_PRESS,glfwGetFramebufferSize
from plugin import Plugin

import logging
logger = logging.getLogger(__name__)

class Trim_Marks(Plugin):
    """docstring for Trim_Mark
    """
    def __init__(self, g_pool, focus=0, sections=[]):
        super(Trim_Marks, self).__init__(g_pool)
        g_pool.trim_marks = self #attach self for ease of access by others.
        self.order = .8
        self.capture = g_pool.capture
        self.frame_count = self.capture.get_frame_count()

        # focused section
        self._focus = focus

        # sections
        if sections:
            self._sections = sections
            self._in_mark, self._out_mark = self._sections[self._focus]
        else:
            self._in_mark = 0
            self._out_mark = self.frame_count
            sections.append([self._in_mark, self._out_mark])
            self._sections = sections

        self.mid_sections = [self.get_mid_section(s)for s in self._sections] 

        self.drag_in = False
        self.drag_out = False

        #display layout
        self.padding = 20. #in screen pixel
        self.window_size = 0,0

        self.frame_size = self.capture.frame_size

    @property
    def sections(self):
        return self._sections

    @sections.setter
    def sections(self, value):
        self._sections = value
        self.mid_sections = [self.get_mid_section(s) for s in self._sections]       

    @property
    def focus(self):
        return self._focus

    @focus.setter
    def focus(self, value):
        self._focus = value
        (self._in_mark, self._out_mark) = self.sections[self._focus]

    @property
    def in_mark(self):
        return self._in_mark

    @in_mark.setter
    def in_mark(self, value):
        self._in_mark = int(min(self._out_mark,max(0,value)))
        self.sections[self.focus][0] = self._in_mark
        self.mid_sections[self.focus] = self.get_mid_section(self.sections[self.focus])

    @property
    def out_mark(self):
        return self._out_mark

    @out_mark.setter
    def out_mark(self, value):
        self._out_mark = int(min(self.frame_count,max(self.in_mark,value)))
        self.sections[self.focus][1] = self._out_mark
        self.mid_sections[self.focus] = self.get_mid_section(self.sections[self.focus])

    def get_mid_section(self, s):
        return int(s[0] + ((s[1]-s[0])/2))

    def set(self,mark_range):
        self._in_mark,self._out_mark = mark_range
        self.sections[self.focus][0] = self._in_mark
        self.sections[self.focus][1] = self._out_mark
        self.mid_sections[self.focus] = self.get_mid_section(self.sections[self.focus])

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

        elif self.drag_out:
            x,y = glfwGetCursorPos(glfwGetCurrentContext())
            x,_ = self.screen_to_bar_space((x,y))
            self.out_mark = x



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
            if self.drag_out or self.drag_in:
                logger.info("Section: "+self.get_string())
                self.drag_out = False
                self.drag_in = False


            # would be great to expand the click area horizontally for big sections
            for s in self.sections:
                if s is not self.sections[self.focus]:
                    midsec = self.mid_sections[self.sections.index(s)]
                    screen_midsec_pos = self.bar_space_to_screen((midsec,0))
                    dist = abs(pos[0]-screen_midsec_pos[0])+abs(pos[1]-screen_midsec_pos[1])
                    if dist < 10:
                        if self.distance_in_pix(midsec,self.capture.get_frame_index()) > 20:
                            self.focus = self.sections.index(s)
                            break


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
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()

        glOrtho(-self.h_pad,  (self.frame_count)+self.h_pad, -self.v_pad, 1+self.v_pad,-1,1) # ranging from 0 to cache_len-1 (horizontal) and 0 to 1 (vertical)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        color1 = RGBA(.1,.9,.2,.5)
        color2 = RGBA(.1,.9,.9,.5)

        if self.in_mark != 0 or self.out_mark != self.frame_count:
            draw_polyline( [(self.in_mark,0),(self.out_mark,0)],color=color1,thickness=2)
 
        draw_points([(self.in_mark,0),],color=color1,size=10)
        draw_points([(self.out_mark,0),],color=color1,size=10)

        if self.sections:
            for s in self.sections:
                if self.sections.index(s) != self.focus:
                    draw_polyline( [(s[0],0),(s[1],0)],color=RGBA(.1,.9,.9,.2),thickness=2)
                for mark in s:
                    draw_points([(mark,0),],color=color2,size=5)

        if self.mid_sections:
            for m in self.mid_sections:
                draw_points([(m,0),],color=RGBA(.1,.9,.9,.1),size=10)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
