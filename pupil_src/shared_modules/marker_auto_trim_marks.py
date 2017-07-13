'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import logging
logger = logging.getLogger(__name__)

from plugin import Plugin
from offline_surface_tracker import Offline_Surface_Tracker
from video_export_launcher import Video_Export_Launcher
from ctypes import c_int

from pyglui import ui
from gl_utils import adjust_gl_view,clear_gl_screen,basic_gl_setup,cvmat_to_glmat
from pyglui.cygl.utils import RGBA,draw_points,draw_polyline
from OpenGL.GL import *
from glfw import *

import numpy as np
from itertools import groupby


class Marker_Auto_Trim_Marks(Plugin):
    """docstring for Marker_Auto_Trim_Marks:

    Using markers and this plugin sections within one recording can be sliced and autoexported:

    Show marker #x for more than 3 seconds to mark the beginning of a export section
    Show marker #y for more than 3 seconds to mark the end of an export section

    Marker presence is determined by the visibility of a marker for more than 50percent withing 6 seconds.

    This plugin depends on the offline marker tracker plugin to be loaded.

    """

    def __init__(self, g_pool,man_in_marks=[],man_out_marks=[]):
        super().__init__(g_pool)
        self.menu = None

        self.in_marker_id = 18
        self.out_marker_id = 25
        self.active_section = 0
        self.sections = None
        self.gl_display_ranges = []

        self.man_in_marks = man_in_marks
        self.man_out_marks = man_out_marks

        self.video_export_queue = []
        self.surface_export_queue = []
        self.current_frame_idx = 0

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Marker Auto Trim Marks')
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Info_Text("Marker Auto uses the marker detector to get markers"))
        self.menu.append(ui.Button('remove',self.unset_alive))

        #set up bar display padding
        self.on_window_resize(glfwGetCurrentContext(),*glfwGetWindowSize(glfwGetCurrentContext()))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def on_window_resize(self,window,w,h):
        self.win_size = w,h

    def unset_alive(self):
        self.alive = False

    def add_manual_in_mark(self):
        self.man_in_marks.append(self.current_frame_idx)
        self.sections = None

    def del_man_in_mark(self, mark):
        if mark == "select one":
            return
        self.man_in_marks.remove(mark)
        self.sections = None

    def add_manual_out_mark(self):
        self.man_out_marks.append(self.current_frame_idx)
        self.sections = None

    def del_man_out_mark(self, mark):
        if mark == "select one":
            return
        self.man_out_marks.remove(mark)
        self.sections = None

    def enqueue_video_export(self):
        self.video_export_queue = self.sections[:]

    def enqueue_surface_export(self):
        self.surface_export_queue = self.sections[:]

    def video_export(self,section):
        plugins = [p for p in self.g_pool.plugins if isinstance(p,Video_Export_Launcher)]
        if plugins:
            launcher = plugins[0]
            logger.info("exporting {!s}".format(section))
            self.g_pool.trim_marks.set(section)
            launcher.rec_name.value = "world_viz_section_{}-{}".format(*section)
            launcher.add_export()

    def surface_export(self,section):
        plugins = [p for p in self.g_pool.plugins if isinstance(p,Offline_Surface_Tracker)]
        if plugins:
            tracker = plugins[0]
            logger.info("exporting {!s}".format(section))
            self.g_pool.trim_marks.set(section)
            tracker.recalculate()
            tracker.save_surface_statsics_to_file()
        else:
            logger.warning("Please start Offline_Surface_Tracker Plugin for surface export.")

    def activate_section(self,section):
        self.g_pool.trim_marks.set(section)
        self.active_section = section

    def get_init_dict(self):
        d = {'man_out_marks':self.man_out_marks,'man_in_marks':self.man_in_marks}
        return d

    def update_bar_indicator(self,status):
        if status:
            self.menu[0].text = "Marker Auto uses the marker detector to get markers"
        else:
            self.menu[0].text  = "Marker Auto Trim Marks: Turn on Offline_Surface_Tracker!"


    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        self.current_frame_idx = frame.index
        if self.video_export_queue:
            self.video_export(self.video_export_queue.pop(0))
        if self.surface_export_queue:
            self.surface_export(self.surface_export_queue.pop(0))

        if self.sections == None:
            plugins = [p for p in self.g_pool.plugins if isinstance(p,Offline_Surface_Tracker)]
            if plugins:
                marker_tracker_plugin = plugins[0]
            else:
                self.update_bar_indicator(False)
                return

            if marker_tracker_plugin.cache.complete:
                #make a marker signal 0 = none, 1 = in, -1=out
                in_id = self.in_marker_id
                out_id = self.out_marker_id
                logger.debug("Looking for trim mark markers: {},{}".format(in_id, out_id))
                in_out_signal = [0]*len(marker_tracker_plugin.cache)
                for idx,frame in enumerate(marker_tracker_plugin.cache):
                    # marker = {'id':msg,'verts':r,'verts_norm':r_norm,'centroid':centroid,"frames_since_true_detection":0}
                    for marker in frame:
                        if marker['id'] == in_id:
                            in_out_signal[idx] +=1
                        if marker['id'] == out_id:
                            in_out_signal[idx] -=1

                # make a smooth signal
                in_out_smooth = np.convolve(in_out_signal,[2./30]*30,mode='same') #mean filter with sum 2 and len 60,
                #Mode 'same' returns output of length max(signal, filter).

                # find falling edges of in markers clusters
                i = -1
                in_falling_edge_idx = []
                for t,g in groupby(in_out_smooth,lambda x:x>=1):
                    s = i + 1
                    i += len(list(g))
                    if t:
                        in_falling_edge_idx.append(i)

                # find rising edges of out markers clusters
                i = -1
                out_rising_edge_idx = []
                for t,g in groupby(in_out_smooth,lambda x:x<=-1):
                    s = i + 1
                    i += len(list(g))
                    if t:
                        out_rising_edge_idx.append(s)

                events = [('out',idx)for idx in out_rising_edge_idx]+[('in',idx)for idx in in_falling_edge_idx]
                manual_markers = [('in',idx) for idx in self.man_in_marks] + [('out',idx) for idx in self.man_out_marks]
                events += manual_markers
                events.sort(key=lambda x: x[1])
                events = [('in',0)]+ events + [('out',len(marker_tracker_plugin.cache))]

                self.sections = []
                for t,g in groupby(events,lambda x:x[0]):
                    if t == "in":
                        last_in_marker_of_this_cluster = list(g)[-1]
                        section_in_index = last_in_marker_of_this_cluster[1]
                    else:
                        #t=="out"
                        fist_out_marker_of_this_clutser = g.next() #first item in cluster
                        section_out_index = fist_out_marker_of_this_clutser[1]
                        self.sections.append((section_in_index,section_out_index))

                self.sections = [(s, e) for s, e in self.sections if e-s>10]#we filter out tiny sections
                # because they can happen with out markers at video start and in marker at video end.

                # Lines for areas that have been cached
                self.gl_display_ranges = []
                for r in self.sections: # [[0,1],[3,4]]
                    self.gl_display_ranges += (r[0],0),(r[1],0) #[(0,0),(1,0),(3,0),(4,0)]

                if self.sections:
                    self.activate_section=self.sections[0]
                self.menu.elements[:] = []
                self.menu.append(ui.Button('remove',self.unset_alive))
                self.menu.label
                self.menu.append(ui.Slider('in_marker_id',self,min=0,step=1,max=63,label='IN marker id'))
                self.menu.append(ui.Slider('out_marker_id',self,min=0,step=1,max=63,label='OUT marker id'))
                self.menu.append(ui.Selector('active_section',self,selection=self.sections,setter=self.activate_section,label='set section'))
                self.menu.append(ui.Button('video export all sections',self.enqueue_video_export))
                self.menu.append(ui.Button('surface export all sections',self.enqueue_surface_export))

                self.menu.append(ui.Button('add in_mark here',self.add_manual_in_mark))
                self.menu.append(ui.Selector('man_in_mark',selection=self.man_in_marks,setter=self.del_man_in_mark,getter=lambda:"select one",label='del manual in marker'))

                self.menu.append(ui.Button('add out mark here',self.add_manual_out_mark))
                self.menu.append(ui.Selector('man_out_mark',selection=self.man_out_marks,setter=self.del_man_out_mark,getter=lambda:"select one",label='del manual out marker'))
            else:
                self.menu.label = "Marker Auto Trim Marks: Waiting for Cacher to finish"

    def gl_display(self):
        if self.sections:
            self.gl_display_cache_bars()

    def gl_display_cache_bars(self):
        """
        """
        padding = 20.
        frame_max = len(self.g_pool.timestamps) #last marker is garanteed to be frame max.

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        width,height = self.win_size
        h_pad = padding * (frame_max-2)/float(width)
        v_pad = padding* 1./(height-2)
        gluOrtho(-h_pad,  (frame_max-1)+h_pad, -v_pad, 1+v_pad,-1, 1) # ranging from 0 to cache_len-1 (horizontal) and 0 to 1 (vertical)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(0,-.02,0)
        color = (7.,.1,.2,8.)
        draw_polyline(self.gl_display_ranges,color=RGBA(*color),line_type=GL_LINES,thickness=2.)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def cleanup(self):
        self.deinit_gui()
