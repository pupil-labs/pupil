'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys, os,platform
import cv2
import numpy as np
from file_methods import Persistent_Dict
from gl_utils import draw_gl_polyline,adjust_gl_view,draw_gl_polyline_norm,clear_gl_screen,draw_gl_point,draw_gl_points,draw_gl_point_norm,draw_gl_points_norm,basic_gl_setup,cvmat_to_glmat, draw_named_texture
from methods import normalize,denormalize
from glfw import *
import atb
from ctypes import c_int,c_bool,create_string_buffer,c_float

from plugin import Plugin
#logging
import logging
logger = logging.getLogger(__name__)

from square_marker_detect import detect_markers_robust,detect_markers_simple, draw_markers,m_marker_to_screen
from reference_surface import Reference_Surface
from math import sqrt

class Marker_Detector(Plugin):
    """docstring
    """
    def __init__(self,g_pool,menu_conf={}):
        super(Marker_Detector, self).__init__(g_pool)
        self.g_pool = g_pool
        self.order = .2

        # all markers that are detected in the most recent frame
        self.markers = []
        # all registered surfaces

        self.K = np.load(os.path.join(self.g_pool.user_dir,'camera_matrix.npy'))
        self.dist_coef = np.load(os.path.join(self.g_pool.user_dir,"dist_coefs.npy"))
        self.img_size = np.load(os.path.join(self.g_pool.user_dir,"camera_resolution.npy"))

        self.surface_definitions = Persistent_Dict(os.path.join(g_pool.user_dir,'surface_definitions') )
        self.surfaces = [Reference_Surface(saved_definition=d, camera_intrinsics=(self.K, self.dist_coef, self.img_size)) for d in self.load('realtime_square_marker_surfaces',[]) if isinstance(d,dict)]

        # edit surfaces
        self.surface_edit_mode = c_bool(0)
        self.edit_surfaces = []

        #detector vars
        self.robust_detection = c_bool(1)
        self.aperture = c_int(11)
        self.min_marker_perimeter = 80
        self.locate_3d = c_bool(0)

        #debug vars
        self.draw_markers = c_bool(0)
        self.show_surface_idx = c_int(0)
        self.recent_pupil_positions = []

        self.img_shape = None

        atb_label = "marker detection"
        self._bar = atb.Bar(name =self.__class__.__name__, label=atb_label,
            help="marker detection parameters", color=(50, 150, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 300))

        self.update_bar_markers()



    def unset_alive(self):
        self.alive = False

    def load(self, var_name, default):
        return self.surface_definitions.get(var_name,default)
    def save(self, var_name, var):
            self.surface_definitions[var_name] = var


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

    def add_surface(self):
        self.surfaces.append(Reference_Surface(camera_intrinsics=(self.K, self.dist_coef, self.img_size)))
        self.update_bar_markers()

    def remove_surface(self,i):
        self.surfaces[i].cleanup()
        del self.surfaces[i]
        self.update_bar_markers()

    def update_bar_markers(self):
        self._bar.clear()
        self._bar.add_button('close',self.unset_alive)
        self._bar.add_var('robust_detection',self.robust_detection)
        self._bar.add_var("3D location", self.locate_3d)
        self._bar.add_var("draw markers",self.draw_markers)
        self._bar.add_button("  add surface   ", self.add_surface, key='a')
        self._bar.add_var("  edit mode   ", self.surface_edit_mode )
        for s,i in zip(self.surfaces,range(len(self.surfaces)))[::-1]:
            self._bar.add_var("%s_window"%i,setter=s.toggle_window,getter=s.window_open,group=str(i),label='open in window')
            self._bar.add_var("%s_name"%i,create_string_buffer(512),getter=s.atb_get_name,setter=s.atb_set_name,group=str(i),label='name')
            self._bar.add_var("%s_markers"%i,create_string_buffer(512), getter=s.atb_marker_status,group=str(i),label='found/registered markers' )
            self._bar.add_var("%s_x_scale"%i,vtype=c_float, getter=s.atb_get_scale_x, min=1,setter=s.atb_set_scale_x,group=str(i),label='real width', help='this scale factor is used to adjust the coordinate space for your needs (think photo pixels or mm or whatever)' )
            self._bar.add_var("%s_y_scale"%i,vtype=c_float, getter=s.atb_get_scale_y,min=1,setter=s.atb_set_scale_y,group=str(i),label='real height',help='defining x and y scale factor you atumatically set the correct aspect ratio.' )
            self._bar.add_button("%s_remove"%i, self.remove_surface,data=i,label='remove',group=str(i))

    def update(self,frame,recent_pupil_positions,events):
        gray = frame.gray
        self.img_shape = frame.height,frame.width,3

        if self.robust_detection.value:
            self.markers = detect_markers_robust(gray,
                                                grid_size = 5,
                                                prev_markers=self.markers,
                                                min_marker_perimeter=self.min_marker_perimeter,
                                                aperture=self.aperture.value,
                                                visualize=0,
                                                true_detect_every_frame=3)
        else:
            self.markers = detect_markers_simple(gray,
                                                grid_size = 5,
                                                min_marker_perimeter=self.min_marker_perimeter,
                                                aperture=self.aperture.value,
                                                visualize=0)

        # locate surfaces
        for s in self.surfaces:
            s.locate(self.markers, self.locate_3d.value)
            # if s.detected:
                # events.append({'type':'marker_ref_surface','name':s.name,'uid':s.uid,'m_to_screen':s.m_to_screen,'m_from_screen':s.m_from_screen, 'timestamp':frame.timestamp})

        if self.draw_markers.value:
            draw_markers(frame.img,self.markers)

        # edit surfaces by user
        if self.surface_edit_mode:
            window = glfwGetCurrentContext()
            pos = glfwGetCursorPos(window)
            pos = normalize(pos,glfwGetWindowSize(window))
            pos = denormalize(pos,(self.img_shape[1],self.img_shape[0]) ) # Position in img pixels

            for s,v_idx in self.edit_surfaces:
                if s.detected:
                    pos = normalize(pos,(self.img_shape[1],self.img_shape[0]),flip_y=True)
                    new_pos =  s.img_to_ref_surface(np.array(pos))
                    s.move_vertex(v_idx,new_pos)

        #map recent gaze onto detected surfaces used for pupil server
        for s in self.surfaces:
            if s.detected:
                s.gaze_on_srf = []
                for p in recent_pupil_positions:
                    if p['norm_pupil'] is not None:
                        gp_on_s = tuple(s.img_to_ref_surface(np.array(p['norm_gaze'])))
                        p['realtime gaze on '+s.name] = gp_on_s
                        s.gaze_on_srf.append(gp_on_s)


        #allow surfaces to open/close windows
        for s in self.surfaces:
            if s.window_should_close:
                s.close_window()
            if s.window_should_open:
                s.open_window()


    def gl_display(self):
        """
        Display marker and surface info inside world screen
        """
        for m in self.markers:
            hat = np.array([[[0,0],[0,1],[.5,1.3],[1,1],[1,0],[0,0]]],dtype=np.float32)
            hat = cv2.perspectiveTransform(hat,m_marker_to_screen(m))
            draw_gl_polyline(hat.reshape((6,2)),(0.1,1.,1.,.5))

        for s in self.surfaces:
            s.gl_draw_frame()

            if self.locate_3d.value:
                s.gl_display_in_window_3d(self.g_pool.image_tex)
            else:
                s.gl_display_in_window(self.g_pool.image_tex)


        if self.surface_edit_mode.value:
            for s in  self.surfaces:
                s.gl_draw_corners()


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self.save("realtime_square_marker_surfaces",[rs.save_to_dict() for rs in self.surfaces if rs.defined])

        self.surface_definitions.close()

        for s in self.surfaces:
            s.close_window()
        self._bar.destroy()
