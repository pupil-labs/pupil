'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from os import path
from ast import literal_eval

from pyglui.cygl.utils import RGBA,draw_points,draw_polyline
from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D
from glfw import glfwGetWindowSize, glfwGetCurrentContext, GLFW_KEY_V, GLFW_KEY_COMMA
from pyglui import ui
import numpy as np

from plugin import Plugin

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) 

class Segmentation(Plugin):
    """
    The user can manually create events by pressing
    keyboard keys.

    This plugin will display vertical bars at the bottom seek bar
    based on those events.

    One should be able send those events
    as sections to the trim_marks plugin (auto-trim).

    The auto-trim functionality include two options:
    - example events = list(1, 50, 100, 150)
    - chain
      - would return the following sections[(1,50), (51,100), (101, 150)]
    - in out pairs
      - would return the following sections[(1,50), (100, 150)]
  
    Todo:
    - import events from Pupil like timestamps
    - selector to manage multiple saved sections

    """
    def __init__(self, g_pool, custom_events=[], mode='chain', keep_create_order=True):
        super(Segmentation, self).__init__(g_pool)
        
        # Pupil Player system configs
        self.trim_marks = g_pool.trim_marks
        self.order = .8
        self.uniqueness = "unique"

        # Pupil Player data
        self.capture = g_pool.capture
        #self.current_frame_index = self.capture.get_frame_index()
        self.frame_count = self.capture.get_frame_count()
        self.frame_index = None
        # self.timestamps = g_pool.timestamps

        # display layout
        self.padding = 20. #in screen pixel

        # initialize empty menu and local variables
        self.menu = None
        self.mode = mode
        self.keep_create_order = keep_create_order
  
        # persistence
        self.custom_events_path = path.join(self.g_pool.rec_dir,'custom_events.npy')
        try:
            self.custom_events = list(np.load(self.custom_events_path))
            logger.info("Custom events were found at: "+ self.custom_events_path)
        except:
            logger.warning("No custom events were found at: "+ self.custom_events_path)
            self.custom_events = custom_events
            if not self.custom_events:
                logger.warning("No chached events were found.")
            else:
                logger.warning("Using chached events. Please, save them if necessary. Otherwise, if you close Segmentation plugin those events will be lost.")
 
    def event_undo(self, arg):
        if self.custom_events:
            self.custom_events.pop()
            if not self.keep_create_order:
                self.custom_events = sorted(self.custom_events, key=int)
  
    def create_custom_event(self, arg):
        if self.frame_index:
            if self.frame_index not in self.custom_events:
                self.custom_events.append(self.frame_index)
                if not self.keep_create_order:
                    self.custom_events = sorted(self.custom_events, key=int)

    def save_custom_event(self):  
        np.save(self.custom_events_path,np.asarray(self.custom_events))

    def auto_trim(self):
        # create sections and pass them to the trim_marks
        sections = []
        events = sorted(self.custom_events, key=int)
        size = len(events)
        if size > 1:
            i = 0
            while True:
                if self.mode == 'chain':
                    if i == 0:
                        sections.append([events[i],events[i+1]])
                    elif (i > 0) and (i < (size-1)):
                        sections.append([events[i]+1,events[i+1]])
                    i += 1
                
                elif self.mode == 'in out pairs':
                    if i < (size-1):
                        sections.append([events[i],events[i+1]])
                    i += 2

                if i > (size-1):
                    break

        self.trim_marks.sections = sections
        self.trim_marks.focus = 0

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Segmentation')
        # add ui elements to the menu
        self.menu.append(ui.Button('Close', self.unset_alive))
        self.menu.append(ui.Info_Text('You can create custom events by pressing "v". To undo press ", (comma)". Remember to save them when your were done.'))
        self.menu.append(ui.Switch('keep_create_order',self,label="Keep Creation Order"))
        # maybe thumbs instead keyboard keys?
        self.menu.append(ui.Hot_Key('create_event',setter=self.create_custom_event,getter=lambda:True,label='V',hotkey=GLFW_KEY_V))
        self.menu.append(ui.Hot_Key('event_undo',setter=self.event_undo,getter=lambda:True,label=',',hotkey=GLFW_KEY_COMMA))
        self.menu.append(ui.Button('Save Events',self.save_custom_event))
        self.menu.append(ui.Info_Text('You can auto-trim based on avaiable events. Choose the Trim Mode that fit your needs.'))
        self.menu.append(ui.Selector('mode',self,label='Trim Mode',selection=['chain','in out pairs'] )) 
        self.menu.append(ui.Button('Auto-trim',self.auto_trim))

        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self.on_window_resize(glfwGetCurrentContext(),*glfwGetWindowSize(glfwGetCurrentContext()))

    def cleanup(self):
        self.deinit_gui()

    def on_window_resize(self,window,w,h):
        self.window_size = w,h
        self.h_pad = self.padding * self.frame_count/float(w)
        self.v_pad = self.padding * 1./h

    def update(self,frame,events):
        self.frame_index = frame.index

    def gl_display(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(-self.h_pad,  (self.frame_count)+self.h_pad, -self.v_pad, 1+self.v_pad) # ranging from 0 to cache_len-1 (horizontal) and 0 to 1 (vertical)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # custom events
        for e in self.custom_events:
            draw_polyline([(e,.06),(e,.005)], color = RGBA(.8, .8, .8, .8))

        size = len(self.custom_events)
        if size > 1:
            for i, e in enumerate(self.custom_events):
                draw_points([(e, .03)], size = 5, color = RGBA(.1, .5, .5, 1.)) 

            i = 0
            while True:
                if i == 0:
                    draw_polyline([(self.custom_events[i],.03),(self.custom_events[i+1],0.03)], color = RGBA(.8, .8, .8, .8))
                elif (i > 0) and (i < (size-1)):
                    draw_polyline([(self.custom_events[i] +1,.03),(self.custom_events[i+1],0.03)], color = RGBA(.8, .8, .8, .8))

                if 'chain' in self.mode:
                    i += 1
                elif 'in out pairs' in self.mode:
                    i += 2

                if i > (size-1):
                    break

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def get_init_dict(self):
        return {'custom_events':self.custom_events,
                'mode':self.mode,
                'keep_create_order':self.keep_create_order}

    def clone(self):
        return Segmentation(**self.get_init_dict())