'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import os
from pyglui import ui
from plugin import Plugin
from file_methods import save_object, load_object

import numpy as np
from OpenGL.GL import *
from glfw import glfwGetWindowSize,glfwGetCurrentContext
from pyglui.cygl.utils import draw_polyline,RGBA
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

#logging
import logging
logger = logging.getLogger(__name__)

class Annotation_Capture(Plugin):
    """Describe your plugin here
    """
    def __init__(self,g_pool,annotations=[('My annoation','e')]):
        super(Annotation_Capture, self).__init__(g_pool)
        self.menu = None
        self.sub_menu = None
        self.buttons = []

        self.annotations = annotations[:]

        self.new_annotation_name = 'new annotation name'
        self.new_annotation_hotkey = 'e'


    def init_gui(self):
        #lets make a menu entry in the sidebar
        self.menu = ui.Growing_Menu('User Defined Events')
        self.g_pool.sidebar.append(self.menu)

        #add a button to close the plugin
        self.menu.append(ui.Button('close User_Events',self.close))
        self.menu.append(ui.Text_Input('new_annotation_name',self))
        self.menu.append(ui.Text_Input('new_annotation_hotkey',self))
        self.menu.append(ui.Button('add annotation type',self.add_annotation))
        self.sub_menu = ui.Growing_Menu('Events - click to remove')
        self.menu.append(self.sub_menu)
        self.update_buttons()


    def update_buttons(self):
        for b in self.buttons:
            self.g_pool.quickbar.remove(b)
            self.sub_menu.elements[:] = []
        self.buttons = []

        for e_name,hotkey in self.annotations:

            def make_fire(e_name,hotkey):
                return lambda _ : self.fire_annotation(e_name)

            def make_remove(e_name,hotkey):
                return lambda: self.remove_event((e_name,hotkey))

            thumb = ui.Thumb(e_name,setter=make_fire(e_name,hotkey), getter=lambda: False,
            label=hotkey,hotkey=hotkey)
            self.buttons.append(thumb)
            self.g_pool.quickbar.append(thumb)
            self.sub_menu.append(ui.Button(e_name+" <"+hotkey+">",make_remove(e_name,hotkey)))



    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None
        if self.buttons:
            for b in self.buttons:
                self.g_pool.quickbar.remove(b)
            self.buttons = []

    def add_annotation(self):
        self.annotations.append((self.new_annotation_name,self.new_annotation_hotkey))
        self.update_buttons()

    def remove_annotations(self,annotation):
        self.annotations.remove(annotation)
        self.update_buttons()

    def close(self):
        self.alive = False

    def fire_annotation(self,annotation_label):
        t = self.g_pool.capture.get_timestamp()
        logger.info('"%s"@%s'%(annotation_label,t))
        notification = {'subject':'annotation','label':annotation_label,'timestamp':t,'duration':0.0,'source':'local','network_propagate':True} #you may add more field to this dictionary if you want.
        self.notify_all(notification)


    def get_init_dict(self):
        return {'annotations':self.annotations}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()


class User_Event_Player(Plugin):
    """Describe your plugin here
    When captured file is played, event tags (straight line pertruding
         from bar) should appear along the video bar
    """
    def __init__(self,g_pool):
        super(User_Event_Player, self).__init__(g_pool)
        from player_methods import correlate_data


        self.menu = None
        self.frame_count = len(self.g_pool.timestamps)

        #display layout
        self.padding = 20. #in sceen pixel
        self.window_size = 0,0


        self.events_list = load_object(os.path.join(self.g_pool.rec_dir, "user_events"))
        correlate_data(self.events_list, self.g_pool.timestamps)

        self.event_by_timestamp = dict( [(i['timestamp'],i) for i in self.events_list])
        self.event_by_index = dict( [(i['index'],i) for i in self.events_list])


    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Event Player')

        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Button('remove',self.unset_alive))
        self.on_window_resize(glfwGetCurrentContext(),*glfwGetWindowSize(glfwGetCurrentContext()))

        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans',get_opensans_font_path())
        self.glfont.set_size(24)
        #self.glfont.set_color_float((0.2,0.5,0.9,1.0))
        self.glfont.set_align_string(v_align='center',h_align='middle')


    def on_window_resize(self,window,w,h):
        self.window_size = w,h
        self.h_pad = self.padding * self.frame_count/float(w)
        self.v_pad = self.padding * 1./h


    def update(self,frame,events):
        event = self.event_by_index.get(frame.index,None)
        if event:
            logger.info(event['user_event_name'])

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


        for e in self.events_list:
            logger.info(str(e['index']))

            draw_polyline(verts=[(e['index'],0),(e['index'],.02)],
                    color=RGBA(*event_tick_color))
            self.glfont.set_color_float((1.,0.,0.,.8))
            #self.glfont.set_blur(0.96)
            self.glfont.draw_text(e['index'],0.02,e['user_event_name'])


        # draw_polyline(verts=[(0,0),(self.current_frame_index,0)],color=RGBA(*color1))
        # draw_polyline(verts=[(self.current_frame_index,0),(self.frame_count,0)],color=RGBA(.5,.5,.5,.5))
        # draw_points([(self.current_frame_index,0)],color=RGBA(*color1),size=40)
        # draw_points([(self.current_frame_index,0)],color=RGBA(*color2),size=10)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def get_init_dict(self):
        return {}


    def cleanup(self):
        """called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
