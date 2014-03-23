'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
import shelve
from gl_utils import draw_gl_polyline,adjust_gl_view,draw_gl_polyline_norm,clear_gl_screen,draw_gl_point,draw_gl_points,draw_gl_point_norm,draw_gl_points_norm,basic_gl_setup,cvmat_to_glmat, redraw_gl_texture
from methods import normalize,denormalize
import atb
import audio
from ctypes import c_int,c_bool,create_string_buffer

from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D

from glfw import *
from plugin import Plugin
#logging
import logging
logger = logging.getLogger(__name__)

from square_marker_detect import detect_markers_robust,detect_markers_simple, draw_markers,m_marker_to_screen
from reference_surface import Reference_Surface
from math import sqrt

# window calbacks
def on_resize(window,w, h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)

class Marker_Detector(Plugin):
    """docstring

    """
    def __init__(self,g_pool,atb_pos=(320,220)):
        super(Marker_Detector, self).__init__()
        self.g_pool = g_pool
        self.order = .2

        # all markers that are detected in the most recent frame
        self.markers = []
        # all registered surfaces

        if g_pool.app == 'capture':
            self.surface_definitions = shelve.open(os.path.join(g_pool.user_dir,'surface_definitions'),protocol=2)
            self.surfaces = [Reference_Surface(saved_definition=d) for d in self.load('realtime_square_marker_surfaces',[]) if isinstance(d,dict)]
        elif g_pool.app == 'player':
            #in player we load from the rec_dir: but we have a couple options:
            self.surface_definitions = shelve.open(os.path.join(g_pool.rec_dir,'surface_definitions'),protocol=2)
            if self.load('offline_square_marker_surfaces',[]) != []:
                logger.debug("Found ref surfaces defined or copied in previous session.")
                self.surfaces = [Reference_Surface(saved_definition=d) for d in self.load('offline_square_marker_surfaces',[]) if isinstance(d,dict)]
            elif self.load('offline_square_marker_surfaces',[]) != []:
                logger.debug("Did not find ref surfaces def created or used by the user in player from earlier session. Loading surfaces defined during capture.")
                self.surfaces = [Reference_Surface(saved_definition=d) for d in self.load('realtime_square_marker_surfaces',[]) if isinstance(d,dict)]
            else:
                logger.debug("No surface defs found. Please define using GUI.")
                self.surfaces = []
        # edit surfaces
        self.surface_edit_mode = c_bool(0)
        self.edit_surfaces = []

        #detector vars
        self.robust_detection = c_bool(1)
        self.aperture = c_int(11)
        self.min_marker_perimeter = 80

        #debug vars
        self.draw_markers = c_bool(0)
        self.show_surface_idx = c_int(0)
        self.recent_pupil_positions = []

        self.img_shape = None

        #multi monitor setup
        self.window_should_open = False
        self.window_should_close = False
        self._window = None
        self.fullscreen = c_bool(0)
        self.monitor_idx = c_int(0)
        monitor_handles = glfwGetMonitors()
        self.monitor_names = [glfwGetMonitorName(m) for m in monitor_handles]
        monitor_enum = atb.enum("Monitor",dict(((key,val) for val,key in enumerate(self.monitor_names))))
        #primary_monitor = glfwGetPrimaryMonitor()

        atb_label = "marker detection"
        self._bar = atb.Bar(name =self.__class__.__name__, label=atb_label,
            help="marker detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 100))
        self._bar.add_var("monitor",self.monitor_idx, vtype=monitor_enum,group="Window",)
        self._bar.add_var("fullscreen", self.fullscreen,group="Window")
        self._bar.add_button("  open Window   ", self.do_open, key='m',group="Window")
        self._bar.add_var("surface to show",self.show_surface_idx, step=1,min=0,group="Window")
        self._bar.add_var('robust_detection',self.robust_detection,group="Detector")
        self._bar.add_var("draw markers",self.draw_markers,group="Detector")
        self._bar.add_button('close',self.unset_alive)


        atb_pos = atb_pos[0],atb_pos[1]+110
        self._bar_markers = atb.Bar(name =self.__class__.__name__+'markers', label='registered surfaces',
            help="list of registered ref surfaces", color=(50, 100, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 120))
        self.update_bar_markers()



    def unset_alive(self):
        self.alive = False

    def load(self, var_name, default):
        return self.surface_definitions.get(var_name,default)
    def save(self, var_name, var):
            self.surface_definitions[var_name] = var


    def do_open(self):
        if not self._window:
            self.window_should_open = True

    def on_click(self,pos,button,action):
        if self.surface_edit_mode.value:
            if self.edit_surfaces:
                if action == GLFW_RELEASE:
                    self.edit_surfaces = []
            # no surfaces verts in edit mode, lets see if the curser is close to one:
            else:
                if action == GLFW_PRESS:
                    surf_verts = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
                    x,y = pos
                    for s in self.surfaces:
                        if s.detected:
                            for (vx,vy),i in zip(s.ref_surface_to_img(np.array(surf_verts)),range(4)):
                                vx,vy = denormalize((vx,vy),(self.img_shape[1],self.img_shape[0]),flip_y=True)
                                if sqrt((x-vx)**2 + (y-vy)**2) <15: #img pixels
                                    self.edit_surfaces.append((s,i))

    def advance(self):
        pass

    def open_window(self):
        if not self._window:
            if self.fullscreen.value:
                monitor = glfwGetMonitors()[self.monitor_idx.value]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width= 1280,720

            self._window = glfwCreateWindow(height, width, "Reference Surface", monitor=monitor, share=glfwGetCurrentContext())
            if not self.fullscreen.value:
                glfwSetWindowPos(self._window,200,0)

            on_resize(self._window,height,width)

            #Register callbacks
            glfwSetWindowSizeCallback(self._window,on_resize)
            glfwSetKeyCallback(self._window,self.on_key)
            glfwSetWindowCloseCallback(self._window,self.on_close)

            # gl_state settings
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()

            # refresh speed settings
            glfwSwapInterval(0)


            glfwMakeContextCurrent(active_window)

            self.window_should_open = False


    def on_key(self,window, key, scancode, action, mods):
        if not atb.TwEventKeyboardGLFW(key,int(action == GLFW_PRESS)):
            if action == GLFW_PRESS:
                if key == GLFW_KEY_ESCAPE:
                    self.on_close()

    def on_close(self,window=None):
        self.window_should_close = True

    def close_window(self):
        if self._window:
            glfwDestroyWindow(self._window)
            self._window = None
            self.window_should_close = False

    def add_surface(self):
        self.surfaces.append(Reference_Surface())
        self.update_bar_markers()

    def remove_surface(self,i):
        del self.surfaces[i]
        self.update_bar_markers()


    def update_bar_markers(self):
        self._bar_markers.clear()
        self._bar_markers.add_button("  add surface   ", self.add_surface, key='a')
        self._bar_markers.add_var("  edit mode   ", self.surface_edit_mode )

        for s,i in zip (self.surfaces,range(len(self.surfaces))):
            self._bar_markers.add_var("%s_name"%i,create_string_buffer(512),getter=s.atb_get_name,setter=s.atb_set_name,group=str(i),label='name')
            self._bar_markers.add_var("%s_markers"%i,create_string_buffer(512), getter=s.atb_marker_status,group=str(i),label='found/registered markers' )
            self._bar_markers.add_button("%s_remove"%i, self.remove_surface,data=i,label='remove',group=str(i))

    def update(self,frame,recent_pupil_positions,events):
        img = frame.img
        self.img_shape = frame.img.shape
        if self.robust_detection.value:
            self.markers = detect_markers_robust(img,grid_size = 5,
                                                    prev_markers=self.markers,
                                                    min_marker_perimeter=self.min_marker_perimeter,
                                                    aperture=self.aperture.value,
                                                    visualize=0,
                                                    true_detect_every_frame=3)
        else:
            self.markers = detect_markers_simple(img,grid_size = 5,min_marker_perimeter=self.min_marker_perimeter,aperture=self.aperture.value,visualize=0)

        if self.draw_markers.value:
            draw_markers(img,self.markers)

        # print self.markers

        for s in self.surfaces:
            s.locate(self.markers)
            if s.detected:
                events.append({'type':'marker_ref_surface','name':s.name,'m_to_screen':s.m_to_screen,'m_from_screen':s.m_from_screen, 'timestamp':frame.timestamp})

        if self.surface_edit_mode:
            window = glfwGetCurrentContext()
            pos = glfwGetCursorPos(window)
            pos = normalize(pos,glfwGetWindowSize(window))
            pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels

            for s,v_idx in self.edit_surfaces:
                if s.detected:
                    pos = normalize(pos,(self.img_shape[1],self.img_shape[0]),flip_y=True)
                    new_pos =  s.img_to_ref_surface(np.array(pos))
                    s.move_vertex(v_idx,new_pos)

        #map recent gaze onto detected surfaces used for pupil server
        for p in recent_pupil_positions:
            if p['norm_pupil'] is not None:
                for s in self.surfaces:
                    if s.detected:
                        p['realtime gaze on '+s.name] = tuple(s.img_to_ref_surface(np.array(p['norm_gaze'])))


        if self._window:
            # save a local copy for when we display gaze for debugging on ref surface
            self.recent_pupil_positions = recent_pupil_positions


        if self.window_should_close:
            self.close_window()

        if self.window_should_open:
            self.open_window()

    def gl_display(self):
        """
        Display marker and surface info inside world screen
        """

        for m in self.markers:
            hat = np.array([[[0,0],[0,1],[.5,1.3],[1,1],[1,0],[0,0]]],dtype=np.float32)
            hat = cv2.perspectiveTransform(hat,m_marker_to_screen(m))
            draw_gl_polyline(hat.reshape((6,2)),(0.1,1.,1.,.5))

        for s in  self.surfaces:
            s.gl_draw_frame()

        if self.surface_edit_mode.value:
            for s in  self.surfaces:
                s.gl_draw_corners()


        if self._window and self.surfaces:
            try:
                s = self.surfaces[self.show_surface_idx.value]
            except IndexError:
                s = None
            if s and s.detected:
                self.gl_display_in_window(s)



    def gl_display_in_window(self,surface):
        """
        here we map a selected surface onto a seperate window.
        """
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(self._window)
        clear_gl_screen()

        # cv uses 3x3 gl uses 4x4 tranformation matricies
        m = cvmat_to_glmat(surface.m_from_screen)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, 1, 0, 1) # gl coord convention

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        #apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
        glLoadMatrixf(m)

        #redraw will use the most recent world img texture and use it again.
        redraw_gl_texture( ((0,0),(1,0),(1,1),(0,1)) )

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()


        #now lets get recent pupil positions on this surface:
        try:
            gaze_on_surface = [p['realtime gaze on '+surface.name] for p in self.recent_pupil_positions]
        except KeyError:
            gaze_on_surface = []
        draw_gl_points_norm(gaze_on_surface,color=(0.,8.,.5,.8), size=80)


        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happends either voluntary or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        if self.g_pool.app == 'capture':
            self.save("realtime_square_marker_surfaces",[rs.save_to_dict() for rs in self.surfaces if rs.defined])
        elif self.g_pool.app == 'player':
            self.save("offline_square_marker_surfaces",[rs.save_to_dict() for rs in self.surfaces if rs.defined])
        self.surface_definitions.close()

        if self._window:
            self.close_window()
        self._bar.destroy()
        self._bar_markers.destroy()

