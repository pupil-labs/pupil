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
from gl_utils import draw_gl_polyline,adjust_gl_view,draw_gl_polyline_norm,clear_gl_screen,draw_gl_point,draw_gl_points,draw_gl_point_norm,draw_gl_points_norm,cvmat_to_glmat, draw_named_texture
from pyglui import ui
from methods import normalize,denormalize
from glfw import *
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
        self.order = .2

        # all markers that are detected in the most recent frame
        self.markers = []

        #load camera intrinsics

        try:
            K = np.load(os.path.join(self.g_pool.user_dir,'camera_matrix.npy'))
            dist_coef = np.load(os.path.join(self.g_pool.user_dir,"dist_coefs.npy"))
            img_size = np.load(os.path.join(self.g_pool.user_dir,"camera_resolution.npy"))
            self.camera_intrinsics = K, dist_coefs, img_size
        except:
            self.camera_intrinsics = None

        # all registered surfaces
        self.surface_definitions = Persistent_Dict(os.path.join(g_pool.user_dir,'surface_definitions') )
        self.surfaces = [Reference_Surface(saved_definition=d) for d in  self.surface_definitions.get('realtime_square_marker_surfaces',[]) if isinstance(d,dict)]

        # edit surfaces
        self.surface_edit_mode = 0
        self.edit_surfaces = []

        #detector vars
        self.robust_detection = 1
        self.aperture = 11
        self.min_marker_perimeter = 80
        self.locate_3d = False

        #debug vars
        self.draw_markers = 0
        self.show_surface_idx = 0

        self.img_shape = None




    def unset_alive(self):
        self.alive = False



    def on_click(self,pos,button,action):
        if self.surface_edit_mode:
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
        self.surfaces.append(Reference_Surface())
        self.update_gui_markers()

    def remove_surface(self,i):
        self.surfaces[i].cleanup()
        del self.surfaces[i]
        self.update_bar_markers()

    def init_gui():
        self.menu = ui.Growing_Menu('Marker Tracker')
        self.menu.configuration = self.menu_conf
        self.g_pool.sidebar.append(self.menu)
        self.menu.append(ui.Info_Text('This is the info text. It should explain some non obvious things.'))

        self._bar.add_var("3D location", self.locate_3d)
        self._bar.add_var("draw markers",self.draw_markers)
        self._bar.add_button("  add surface   ", self.add_surface, key='a')
        self._bar.add_var("  edit mode   ", self.surface_edit_mode )
        self.menu.append(ui.Switch('robust_detection',self,on_val=True,off_val=False,label='robust marker detection'))
        self.menu.append(ui.Button(('close',self.unset_alive)))


        self.button = ui.Thumb('running',self,setter=self.toggle,label='Record',hotkey='r')
        self.button.on_color[:] = (1,.0,.0,.8)
        self.g_pool.quickbar.append(self.button)

    def update_gui_markers(self):
        pass
        # self._bar.clear()
        # self._bar.add_button('close',self.unset_alive)
        # self._bar.add_var('robust_detection',self.robust_detection)
        # self._bar.add_var("3D location", self.locate_3d)
        # self._bar.add_var("draw markers",self.draw_markers)
        # self._bar.add_button("  add surface   ", self.add_surface, key='a')
        # self._bar.add_var("  edit mode   ", self.surface_edit_mode )
        # for s,i in zip(self.surfaces,range(len(self.surfaces)))[::-1]:
        #     self._bar.add_var("%s_window"%i,setter=s.toggle_window,getter=s.window_open,group=str(i),label='open in window')
        #     self._bar.add_var("%s_name"%i,create_string_buffer(512),getter=s.atb_get_name,setter=s.atb_set_name,group=str(i),label='name')
        #     self._bar.add_var("%s_markers"%i,create_string_buffer(512), getter=s.atb_marker_status,group=str(i),label='found/registered markers' )
        #     self._bar.add_var("%s_x_scale"%i,vtype=c_float getter=s.atb_get_scale_x, min=1,setter=s.atb_set_scale_x,group=str(i),label='real width', help='this scale factor is used to adjust the coordinate space for your needs (think photo pixels or mm or whatever)')
        #     self._bar.add_var("%s_y_scale"%i,vtype=c_float getter=s.atb_get_scale_y,min=1,setter=s.atb_set_scale_y,group=str(i),label='real height',help='defining x and y scale factor you atumatically set the correct aspect ratio.' )
        #     self._bar.add_button("%s_remove"%i, self.remove_surface,data=i,label='remove',group=str(i))

    def update(self,frame,events):
        gray = frame.gray
        self.img_shape = frame.height,frame.width,3

        if self.robust_detection:
            self.markers = detect_markers_robust(gray,
                                                grid_size = 5,
                                                prev_markers=self.markers,
                                                min_marker_perimeter=self.min_marker_perimeter,
                                                aperture=self.aperture,
                                                visualize=0,
                                                true_detect_every_frame=3)
        else:
            self.markers = detect_markers_simple(gray,
                                                grid_size = 5,
                                                min_marker_perimeter=self.min_marker_perimeter,
                                                aperture=self.aperture,
                                                visualize=0)


        if self.draw_markers:
            draw_markers(frame.img,self.markers)


        # locate surfaces
        for s in self.surfaces:
            s.locate(self.markers, self.locate_3d,self.camera_intrinsics)
            # if s.detected:
                # events.append({'type':'marker_ref_surface','name':s.name,'uid':s.uid,'m_to_screen':s.m_to_screen,'m_from_screen':s.m_from_screen, 'timestamp':frame.timestamp})


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
                for p in events.get('gaze',[]):
                    pass #todo: implement





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

            if self.locate_3d:
                s.gl_display_in_window_3d(self.g_pool.image_tex,self.camera_intrinsics)
            else:
                s.gl_display_in_window(self.g_pool.image_tex)


        if self.surface_edit_mode:
            for s in  self.surfaces:
                s.gl_draw_corners()


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self.surface_definitions["realtime_square_marker_surfaces"] = [rs.save_to_dict() for rs in self.surfaces if rs.defined]
        self.surface_definitions.close()

        for s in self.surfaces:
            s.close_window()
        self._bar.destroy()
