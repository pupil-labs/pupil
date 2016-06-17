'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys, os,platform
import cv2
import numpy as np
from file_methods import Persistent_Dict,load_object
from pyglui.cygl.utils import draw_points,draw_polyline,RGBA
from pyglui import ui
from OpenGL.GL import GL_POLYGON
from methods import normalize,denormalize
from glfw import *
from plugin import Plugin
#logging
import logging
logger = logging.getLogger(__name__)

from square_marker_detect import detect_markers_robust,detect_markers, draw_markers,m_marker_to_screen
from reference_surface import Reference_Surface
from calibration_routines.camera_intrinsics_estimation import load_camera_calibration

from math import sqrt

class Surface_Tracker(Plugin):
    """docstring
    """
    def __init__(self,g_pool,mode="Show Markers and Surfaces",min_marker_perimeter = 100):
        super(Surface_Tracker, self).__init__(g_pool)
        self.order = .2

        # all markers that are detected in the most recent frame
        self.markers = []

        self.camera_calibration = load_camera_calibration(self.g_pool)
        self.load_surface_definitions_from_file()

        # edit surfaces
        self.edit_surfaces = []
        self.edit_surf_verts = []
        self.marker_edit_surface = None
        #plugin state
        self.mode = mode
        self.running = True


        self.robust_detection = 1
        self.aperture = 11
        self.min_marker_perimeter = min_marker_perimeter
        self.locate_3d = False

        self.img_shape = None

        self.menu = None
        self.button =  None
        self.add_button = None

    def load_surface_definitions_from_file(self):
        # all registered surfaces
        self.surface_definitions = Persistent_Dict(os.path.join(self.g_pool.user_dir,'surface_definitions') )
        self.surfaces = [Reference_Surface(saved_definition=d) for d in  self.surface_definitions.get('realtime_square_marker_surfaces',[]) if isinstance(d,dict)]


    def on_click(self,pos,button,action):
        if self.mode == 'Show Markers and Surfaces':
            if action == GLFW_PRESS:
                for s in self.surfaces:
                    toggle = s.get_mode_toggle(pos,self.img_shape)
                    if toggle == 'surface_mode':
                        if s in self.edit_surfaces:
                            self.edit_surfaces.remove(s)
                        else:
                            self.edit_surfaces.append(s)
                    elif toggle == 'marker_mode':
                        if self.marker_edit_surface == s:
                            self.marker_edit_surface = None
                        else:
                            self.marker_edit_surface = s

            if action == GLFW_RELEASE:
                self.edit_surf_verts = []

            elif action == GLFW_PRESS:
                surf_verts = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
                x,y = pos
                for s in self.edit_surfaces:
                    if s.detected and s.defined:
                        for (vx,vy),i in zip(s.ref_surface_to_img(np.array(surf_verts)),range(4)):
                            vx,vy = denormalize((vx,vy),(self.img_shape[1],self.img_shape[0]),flip_y=True)
                            if sqrt((x-vx)**2 + (y-vy)**2) <15: #img pixels
                                self.edit_surf_verts.append((s,i))
                                return

                if self.marker_edit_surface:
                    for m in self.markers:
                        if m['perimeter']>=self.min_marker_perimeter:
                            vx,vy = m['centroid']
                            if sqrt((x-vx)**2 + (y-vy)**2) <15:
                                if self.marker_edit_surface.markers.has_key(m['id']):
                                    self.marker_edit_surface.remove_marker(m)
                                else:
                                    self.marker_edit_surface.add_marker(m,self.markers,self.camera_calibration,self.min_marker_perimeter)

    def add_surface(self,_):
        self.surfaces.append(Reference_Surface())
        self.update_gui_markers()

    def remove_surface(self,i):
        remove_surface = self.surfaces[i]
        if remove_surface == self.marker_edit_surface:
            self.marker_edit_surface = None
        if remove_surface in self.edit_surfaces:
            self.edit_surfaces.remove(remove_surface)


        self.surfaces[i].cleanup()
        del self.surfaces[i]
        self.update_gui_markers()

    def init_gui(self):
        self.menu = ui.Growing_Menu('Surface Tracker')
        self.g_pool.sidebar.append(self.menu)

        self.button = ui.Thumb('running',self,label='Track',hotkey='t')
        self.button.on_color[:] = (.1,.2,1.,.8)
        self.g_pool.quickbar.append(self.button)
        self.add_button = ui.Thumb('add_surface',setter=self.add_surface,getter=lambda:False,label='Add surface',hotkey='a')
        self.g_pool.quickbar.append(self.add_button)
        self.update_gui_markers()

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu= None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None
        if self.add_button:
            self.g_pool.quickbar.remove(self.add_button)
            self.add_button = None

    def update_gui_markers(self):

        def close():
            self.alive = False

        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',close))
        self.menu.append(ui.Info_Text('This plugin detects and tracks fiducial markers visible in the scene. You can define surfaces using 1 or more marker visible within the world view by clicking *add surface*. You can edit defined surfaces by selecting *Surface edit mode*.'))
        self.menu.append(ui.Switch('robust_detection',self,label='Robust detection'))
        self.menu.append(ui.Slider('min_marker_perimeter',self,step=1,min=10,max=500))
        self.menu.append(ui.Switch('locate_3d',self,label='3D localization'))
        self.menu.append(ui.Selector('mode',self,label="Mode",selection=['Show Markers and Surfaces','Show marker IDs'] ))
        self.menu.append(ui.Button("Add surface", lambda:self.add_surface('_'),))

        for s in self.surfaces:
            idx = self.surfaces.index(s)
            s_menu = ui.Growing_Menu("Surface %s"%idx)
            s_menu.collapsed=True
            s_menu.append(ui.Text_Input('name',s))
            s_menu.append(ui.Text_Input('x',s.real_world_size,label='X size'))
            s_menu.append(ui.Text_Input('y',s.real_world_size,label='Y size'))
            s_menu.append(ui.Button('Open Debug Window',s.open_close_window))
            #closure to encapsulate idx
            def make_remove_s(i):
                return lambda: self.remove_surface(i)
            remove_s = make_remove_s(idx)
            s_menu.append(ui.Button('remove',remove_s))
            self.menu.append(s_menu)

    def update(self,frame,events):
        self.img_shape = frame.height,frame.width,3

        if self.running:
            gray = frame.gray

            if self.robust_detection:
                self.markers = detect_markers_robust(gray,
                                                    grid_size = 5,
                                                    prev_markers=self.markers,
                                                    min_marker_perimeter=self.min_marker_perimeter,
                                                    aperture=self.aperture,
                                                    visualize=0,
                                                    true_detect_every_frame=3)
            else:
                self.markers = detect_markers(gray,
                                                grid_size = 5,
                                                min_marker_perimeter=self.min_marker_perimeter,
                                                aperture=self.aperture,
                                                visualize=0)


            if self.mode == "Show marker IDs":
                draw_markers(frame.img,self.markers)

        events['surfaces'] = []

        # locate surfaces
        for s in self.surfaces:
            s.locate(self.markers,self.camera_calibration,self.min_marker_perimeter, self.locate_3d)
            if s.detected:
                events['surfaces'].append({'name':s.name,'uid':s.uid,'m_to_screen':s.m_to_screen.tolist(),'m_from_screen':s.m_from_screen.tolist(), 'timestamp':frame.timestamp})

        if self.running:
            self.button.status_text = '%s/%s'%(len([s for s in self.surfaces if s.detected]),len(self.surfaces))
        else:
            self.button.status_text = 'tracking paused'

        if self.mode == 'Show Markers and Surfaces':
            # edit surfaces by user
            if self.edit_surf_verts:
                window = glfwGetCurrentContext()
                pos = glfwGetCursorPos(window)
                pos = normalize(pos,glfwGetWindowSize(window),flip_y=True)
                for s,v_idx in self.edit_surf_verts:
                    if s.detected:
                        new_pos = s.img_to_ref_surface(np.array(pos))
                        s.move_vertex(v_idx,new_pos)

        #map recent gaze onto detected surfaces used for pupil server
        for s in self.surfaces:
            if s.detected:
                s.gaze_on_srf = []
                for p in events.get('gaze_positions',[]):
                    gp_on_s = tuple(s.img_to_ref_surface(np.array(p['norm_pos'])))
                    p['realtime gaze on ' + s.name] = gp_on_s
                    s.gaze_on_srf.append(gp_on_s)


    def get_init_dict(self):
        return {'mode':self.mode,'min_marker_perimeter':self.min_marker_perimeter}


    def gl_display(self):
        """
        Display marker and surface info inside world screen
        """
        if self.mode == "Show Markers and Surfaces":
            for m in self.markers:
                hat = np.array([[[0,0],[0,1],[.5,1.3],[1,1],[1,0],[0,0]]],dtype=np.float32)
                hat = cv2.perspectiveTransform(hat,m_marker_to_screen(m))
                if m['perimeter']>=self.min_marker_perimeter:
                    draw_polyline(hat.reshape((6,2)),color=RGBA(0.1,1.,1.,.5))
                    draw_polyline(hat.reshape((6,2)),color=RGBA(0.1,1.,1.,.3),line_type=GL_POLYGON)
                else:
                    draw_polyline(hat.reshape((6,2)),color=RGBA(0.1,1.,1.,.5))

            for s in self.surfaces:
                if s not in self.edit_surfaces and s is not self.marker_edit_surface:
                    s.gl_draw_frame(self.img_shape)

            for s in self.edit_surfaces:
                s.gl_draw_frame(self.img_shape,highlight=True,surface_mode=True)
                s.gl_draw_corners()

            if self.marker_edit_surface:
                inc = []
                exc = []
                for m in self.markers:
                    if m['perimeter']>=self.min_marker_perimeter:
                        if self.marker_edit_surface.markers.has_key(m['id']):
                            inc.append(m['centroid'])
                        else:
                            exc.append(m['centroid'])
                draw_points(exc,size=20,color=RGBA(1.,0.5,0.5,.8))
                draw_points(inc,size=20,color=RGBA(0.5,1.,0.5,.8))
                self.marker_edit_surface.gl_draw_frame(self.img_shape,color=(0.0,0.9,0.6,1.0),highlight=True,marker_mode=True)

        for s in self.surfaces:
            if self.locate_3d:
                s.gl_display_in_window_3d(self.g_pool.image_tex,self.camera_intrinsics)
            else:
                s.gl_display_in_window(self.g_pool.image_tex)


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.surface_definitions["realtime_square_marker_surfaces"] = [rs.save_to_dict() for rs in self.surfaces if rs.defined]
        self.surface_definitions.close()

        for s in self.surfaces:
            s.cleanup()
        self.deinit_gui()
