'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


import logging
logger = logging.getLogger(__name__)

from plugin import Plugin
from offline_marker_detector import Offline_Marker_Detector
from export_launcher import Export_Launcher
from ctypes import c_int


from gl_utils import draw_gl_polyline,adjust_gl_view,draw_gl_polyline_norm,clear_gl_screen,draw_gl_point,draw_gl_points,draw_gl_point_norm,draw_gl_points_norm,basic_gl_setup,cvmat_to_glmat, draw_named_texture
from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D
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
    def __init__(self, g_pool,gui_settings={'pos':(220,400),'size':(300,100),'iconified':False},man_in_marks=[],man_out_marks=[]):
        super(Marker_Auto_Trim_Marks, self).__init__()
        self.g_pool = g_pool
        self.gui_settings = gui_settings

        self.in_marker_id = c_int(18)
        self.out_marker_id = c_int(25)
        self.active_section = c_int(0)
        self.sections = None
        self.gl_display_ranges = []

        self.man_in_marks = man_in_marks
        self.man_out_marks = man_out_marks

        self.video_export_queue = []
        self.surface_export_queue = []
        self.current_frame_idx = 0

    def init_gui(self):
        import atb
        self.atb_enum = atb.enum
        pos = self.gui_settings['pos']
        atb_label = "Marker Auto Trim Marks"
        self._bar = atb.Bar(name =self.__class__.__name__+str(id(self)), label=atb_label,
            help="trim marks", color=(50, 150, 50), alpha=50,
            text='light', position=pos,refresh=.1, size=self.gui_settings['size'])

        self._bar.iconified = self.gui_settings['iconified']
        self._bar.add_button('close',self.unset_alive)

        #set up bar display padding
        self.on_window_resize(glfwGetCurrentContext(),*glfwGetWindowSize(glfwGetCurrentContext()))


    def on_window_resize(self,window,w,h):
        self.win_size = w,h


    def unset_alive(self):
        self.alive = False

    def add_manual_in_mark(self):
        self.man_in_marks.append(self.current_frame_idx)
        self.sections = None

    def del_man_in_mark(self, mark_index):
        del self.man_in_marks[mark_index]
        self.sections = None

    def add_manual_out_mark(self):
        self.man_out_marks.append(self.current_frame_idx)
        self.sections = None

    def del_man_out_mark(self, mark_index):
        del self.man_out_marks[mark_index]
        self.sections = None

    def enqueue_video_export(self):
        self.video_export_queue = self.sections[:]

    def enqueue_surface_export(self):
        self.surface_export_queue = self.sections[:]


    def video_export(self,section):
        plugins = [p for p in self.g_pool.plugins if isinstance(p,Export_Launcher)]
        if plugins:
            launcher = plugins[0]
            logger.info("exporting %s" %str(section))
            self.g_pool.trim_marks.set(section)
            launcher.rec_name.value = "world_viz_section_%s-%s"%section
            launcher.add_export()


    def surface_export(self,section):
        plugins = [p for p in self.g_pool.plugins if isinstance(p,Offline_Marker_Detector)]
        if plugins:
            tracker = plugins[0]
            logger.info("exporting %s" %str(section))
            self.g_pool.trim_marks.set(section)
            tracker.recalculate()
            tracker.save_surface_statsics_to_file()
        else:
            logger.warning("Please start Offline_Marker_Detector Plugin for surface export.")

    def activate_section(self,section_idx):
        self.active_section.value = section_idx
        in_mark,out_mark = self.sections[section_idx]
        self.g_pool.trim_marks.set((in_mark,out_mark))


    def set_in_marker(self,m):
        self.in_marker_id.value = m
        self.sections = None

    def set_out_marker(self,m):
        self.out_marker_id.value = m
        self.sections = None


    def get_init_dict(self):
        d = {'man_out_marks':self.man_out_marks,'man_in_marks':self.man_in_marks}
        if hasattr(self,'_bar'):
            gui_settings = {'pos':self._bar.position,'size':self._bar.size,'iconified':self._bar.iconified}
            d['gui_settings'] = gui_settings

        return d

    def update_bar_indicator(self,status):
        if status:
            self._bar.color = (50, 50, 50)
            self._bar.label = "Marker Auto Trim Marks"
        else:
            self._bar.color = (250, 50, 50)
            self._bar.label = "Marker Auto Trim Marks: Turn on Offline_Marker_Detector!"


    def update(self,frame,recent_pupil_positions,events):

        self.current_frame_idx = frame.index
        if self.video_export_queue:
            self.video_export(self.video_export_queue.pop(0))
        if self.surface_export_queue:
            self.surface_export(self.surface_export_queue.pop(0))


        if self.sections == None:
            plugins = [p for p in self.g_pool.plugins if isinstance(p,Offline_Marker_Detector)]
            if plugins:
                marker_tracker_plugin = plugins[0]

            else:
                self.update_bar_indicator(False)
                self._bar.color = (250, 50, 50)
                self._bar.label = "Marker Auto Trim Marks: Turn on Offline_Marker_Detector!"


                return

            if marker_tracker_plugin.cache.complete:


                #make a marker signal 0 = none, 1 = in, -1=out
                in_id = self.in_marker_id.value
                out_id = self.out_marker_id.value
                logger.debug("Looking for trim mark markers: %s,%s"%(in_id,out_id))

                in_out_signal = [0]*len(marker_tracker_plugin.cache)
                for idx,frame in enumerate(marker_tracker_plugin.cache):
                    # marker = {'id':msg,'verts':r,'verts_norm':r_norm,'centroid':centroid,"frames_since_true_detection":0}
                    for marker in frame:
                        if marker['id'] == in_id:
                            in_out_signal[idx] +=1
                        if marker['id'] == out_id:
                            in_out_signal[idx] -=1


                # make a smooth singal
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

                self.sections = [(s,e) for s,e in self.sections if e-s>10]#we filter out tiny sections
                # because they can happen with out markers at video start and in marker at video end.


                # Lines for areas that have been cached
                self.gl_display_ranges = []
                for r in self.sections: # [[0,1],[3,4]]
                    self.gl_display_ranges += (r[0],0),(r[1],0) #[(0,0),(1,0),(3,0),(4,0)]


                self._bar.clear()
                self._bar.add_button('close',self.unset_alive)
                self._bar.label = "Marker Auto Trim Marks"
                self._bar.color = (50, 50, 50)
                self._bar.add_var('IN marker id',vtype = c_int,max=63,min=0, step =1,setter = self.set_in_marker,getter = lambda : self.in_marker_id.value)
                self._bar.add_var('OUT marker id',vtype = c_int, max=63,min=0,step=1, setter = self.set_out_marker,getter = lambda : self.out_marker_id.value)
                self._bar.sections_enum = self.atb_enum("section",dict([(str(r),idx) for idx,r in enumerate(self.sections) ] ))
                self._bar.add_var("set section",vtype=self._bar.sections_enum, getter= lambda: self.active_section.value, setter= self.activate_section)
                self._bar.add_button("video export all sections",self.enqueue_video_export)
                self._bar.add_button("surface export all sections",self.enqueue_surface_export)

                self._bar.add_button("add in_mark here",self.add_manual_in_mark)
                self._bar._del_in_enum = self.atb_enum("del manualin marker",dict([(str(r),idx) for idx,r in enumerate(self.man_in_marks) ] ))
                self._bar.add_var("del manual in mark",vtype=self._bar._del_in_enum, getter= lambda: 0, setter= self.del_man_in_mark)

                self._bar.add_button("add out_mark here",self.add_manual_out_mark)
                self._bar._del_out_enum = self.atb_enum("del manual out marker",dict([(str(r),idx) for idx,r in enumerate(self.man_out_marks) ] ))
                self._bar.add_var("del manual out mark",vtype=self._bar._del_out_enum, getter= lambda: 0, setter= self.del_man_out_mark)

            else:
                self._bar.label = "Marker Auto Trim Marks: Waiting for Cacher to finish"
                self._bar.color = (100, 100, 50)



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
        gluOrtho2D(-h_pad,  (frame_max-1)+h_pad, -v_pad, 1+v_pad) # ranging from 0 to cache_len-1 (horizontal) and 0 to 1 (vertical)


        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(0,-.02,0)
        color = (7.,.1,.2,8.)
        draw_gl_polyline(self.gl_display_ranges,color=color,type='Lines',thickness=2)


        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def cleanup(self):
        self._bar.destroy()
