'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import sys, os, platform
import cv2
import numpy as np
from file_methods import Persistent_Dict,load_object
from pyglui.cygl.utils import draw_points,draw_polyline,RGBA
from glfw import *
from plugin import Plugin
from platform import system

from OpenGL.GL import *
from OpenGL.GL import GL_LINES

from gl_utils import adjust_gl_view, clear_gl_screen, basic_gl_setup, cvmat_to_glmat, make_coord_system_norm_based
from gl_utils.trackball import Trackball

from square_marker_detect import detect_markers,detect_markers_robust, draw_markers,m_marker_to_screen
from collections import deque

# logging
import logging
logger = logging.getLogger(__name__)

class Marker_Tracker_3D(Plugin):
    """
    This plugin tracks the pose of the scene camera based on fiducial markers in the environment.
    """
    def __init__(self,g_pool,min_marker_perimeter = 100,invert_image=False,robust_detection=True):
        super().__init__(g_pool)
        self.visible_markers = []
        self.registered_markers = {}

        self.robust_detection = robust_detection
        self.aperture = 11
        self.min_marker_perimeter = min_marker_perimeter
        self.min_id_confidence = 0.0
        self.invert_image = invert_image

        self.img_shape = None
        self.cam_rot = None
        self.cam_trans = None
        self.scale=2.5
        self.real_world_size = {}
        self.real_world_size['x'] = self.scale # TODO get rid of this
        self.real_world_size['y'] = self.scale

        self.cam_trace = deque(maxlen=200)

        self.name = "Marker Tracker 3D"
        # UI Platform tweaks
        if system() == 'Linux':
            self.window_position_default = (0, 0)
        elif system() == 'Windows':
            self.window_position_default = (8, 31)
        else:
            self.window_position_default = (0, 0)

        self._window = None
        self.fullscreen = False
        self.open_close_window()

    def init_gui(self):
        # self.menu = ui.Growing_Menu('Marker Tracker 3D')
        # self.g_pool.sidebar.append(self.menu)
        #
        # self.button = ui.Thumb('running',self,label='S',hotkey='s')
        # self.button.on_color[:] = (.1,.2,1.,.8)
        # self.g_pool.quickbar.append(self.button)
        # self.add_button = ui.Thumb('add_surface',setter=self.add_surface,getter=lambda:False,label='A',hotkey='a')
        # self.g_pool.quickbar.append(self.add_button)
        # self.update_gui_markers()
        pass

    def deinit_gui(self):
        # if self.menu:
        #     self.g_pool.sidebar.remove(self.menu)
        #     self.menu= None
        # if self.button:
        #     self.g_pool.quickbar.remove(self.button)
        #     self.button = None
        # if self.add_button:
        #     self.g_pool.quickbar.remove(self.add_button)
        #     self.add_button = None
        pass

    def recent_events(self, events):
        # Get current frame
        frame = events.get('frame')
        if not frame:
            return
        self.img_shape = frame.height,frame.width,3

        # Invert frame if necessary
        gray = frame.gray
        if self.invert_image:
            gray = 255-gray

        # Get visible markers and visualize them
        if self.robust_detection:
            self.visible_markers_flipped = detect_markers_robust(
                gray, grid_size = 5,aperture=self.aperture,
                prev_markers=self.visible_markers,
                true_detect_every_frame=3,
                min_marker_perimeter=self.min_marker_perimeter)
        else:
            self.visible_markers_flipped = detect_markers(
                gray, grid_size = 5,aperture=self.aperture,
                min_marker_perimeter=self.min_marker_perimeter)
        # draw_markers(frame.gray,self.visible_markers_flipped)

        # The pixel representation has a flipped y-axis. We repair that for our internal use.
        from copy import deepcopy
        self.visible_markers = deepcopy(self.visible_markers_flipped)
        # for m in self.visible_markers:
        #     for v in m['verts']:
        #         v[0][1] = self.g_pool.capture.frame_size[1] - v[0][1]


        # Process visible markers
        visible_ids = [m['id'] for m in self.visible_markers if m['id_confidence'] >= 0.95]
        if len(self.registered_markers) < 2:
            # Try to perform initial setup
            # Find basis markers

            if 0 in visible_ids and 1 in visible_ids:
                m0 = [m for m in self.visible_markers if m['id'] == 0][0]
                m1 = [m for m in self.visible_markers if m['id'] == 1][0]

                succ0, rot0, trans0 = self.get3DMarkerPos(m0)
                succ1, m_rot, M_trans = self.get3DMarkerPos(m1)

                self.registered_markers[0] = np.array([
                    np.array([0,0,0]),
                    np.array([1,0,0]),
                    np.array([1,1,0]),
                    np.array([0,1,0]),
                ]) * self.scale

                self.registered_markers[1] = np.array([
                    np.array([0, 0, 0]),
                    np.array([1, 0, 0]),
                    np.array([1, 1, 0]),
                    np.array([0, 1, 0]),
                ]) * self.scale
                self.registered_markers[1] = self.registered_markers[1].T
                self.registered_markers[1] = m_rot @ self.registered_markers[1] + M_trans
                self.registered_markers[1] = rot0.T @ (self.registered_markers[1] - trans0)
                self.registered_markers[1] = self.registered_markers[1].T

        else:
            # Compute current camera position from registered visible markers
            # References are markers that are both visible and already registered
            reference_ids = set(visible_ids) & set(self.registered_markers.keys())
            if not  reference_ids:
                print("No reference markers visible!")
            else:
                reference_markers = [m for m in self.visible_markers if m['id'] in reference_ids]

                # Collect marker coordinates in image space and in world space
                img_pos = []
                world_pos = []
                for m in reference_markers:
                    img_pos += m['verts']
                    world_pos += self.registered_markers[m['id']].tolist()
                img_pos = np.asarray(img_pos)
                img_pos.shape = -1,1,2
                world_pos = np.asarray(world_pos)
                world_pos.shape = -1,1,3

                # Compute the camera pose based on the reference markers
                _, self.cam_rot, self.cam_trans = self.g_pool.capture.intrinsics.solvePnP(world_pos, img_pos)
                self.cam_rot = cv2.Rodrigues(self.cam_rot)[0]
                self.cam_trace.append(self.cam_trans.reshape(-1))
                print(self.cam_trans.reshape(-1))


            # Register new markers if references are available
            new_markers = [m for m in self.visible_markers if not m['id'] in reference_ids and m['id_confidence'] >= 0.95]
            if len(reference_ids) > 0 and len(new_markers) > 0:
                for new_m in new_markers:
                    # Make sure marker is far enough away from the borders # TODO check if neccessary
                    v0 = new_m['verts'][0][0]
                    if v0[0] < 200 or v0[0] > 1000 or v0[1] < 200 or v0[1] > 500:
                        continue

                    _, m_rot, M_trans = self.get3DMarkerPos(new_m)

                    verts = np.array([
                        np.array([0, 0, 0]),
                        np.array([1, 0, 0]),
                        np.array([1, 1, 0]),
                        np.array([0, 1, 0]),
                    ]) * self.scale
                    verts = verts.T
                    verts = m_rot @ verts + M_trans
                    verts = self.cam_rot.T @ (verts - self.cam_trans)
                    verts = verts.T

                    self.registered_markers[new_m['id']] = verts


    def get3DMarkerPos(self, marker, target_pts3D=None):

        if target_pts3D is None:
            target_pts = np.array([
                [0.,0.],
                [1.,0.],
                [1.,1.],
                [0.,1.]
            ])* self.scale
            target_pts3D = np.zeros((target_pts.shape[0], target_pts.shape[1] + 1), dtype=np.float32)
            target_pts3D[:, :-1] = target_pts
        else:
            target_pts3D = target_pts3D.copy()
        target_pts3D.shape = -1, 1, 3

        pts = np.asarray(marker['verts'], dtype=np.float32)
        pts.shape = -1, 1, 2

        is3dPoseAvailable, rot3d_cam_to_object, translate3d_cam_to_object = self.g_pool.capture.intrinsics.solvePnP(target_pts3D, pts)

        rot3d_cam_to_object = cv2.Rodrigues(rot3d_cam_to_object)[0]

        return is3dPoseAvailable, rot3d_cam_to_object, translate3d_cam_to_object


    def gl_display(self):
        for m in self.visible_markers_flipped:
            hat = np.array([[[0,0],[0,1],[.5,1.3],[1,1],[1,0],[0,0]]],dtype=np.float32)
            hat = cv2.perspectiveTransform(hat,m_marker_to_screen(m))
            if m['perimeter']>=self.min_marker_perimeter and m['id_confidence']>self.min_id_confidence:
                draw_polyline(hat.reshape((6,2)),color=RGBA(0.1,1.,1.,.5))
                draw_polyline(hat.reshape((6,2)),color=RGBA(0.1,1.,1.,.3),line_type=GL_POLYGON)
            else:
                draw_polyline(hat.reshape((6,2)),color=RGBA(0.1,1.,1.,.5))

        # 3D debug visualization
        self.gl_display_in_window_3d(self.g_pool.image_tex)


    def gl_display_in_window_3d(self,world_tex):
        """
        Display camera pose and markers in 3D space
        """
        K, img_size = self.g_pool.capture.intrinsics.K, self.g_pool.capture.intrinsics.resolution

        if self._window:
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            glClearColor(.8,.8,.8,1.)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearDepth(1.0)
            glDepthFunc(GL_LESS)
            glEnable(GL_DEPTH_TEST)
            self.trackball.push()

            glMatrixMode(GL_MODELVIEW)

            draw_coordinate_system(l=self.real_world_size['x'])

            # Draw registered markers
            visible_ids = [m['id'] for m in self.visible_markers if m['id_confidence'] >= 0.95]
            for id, verts in self.registered_markers.items():
                if id in visible_ids:
                    color = (0, 1, 0, 0.5)
                else:
                    color = (1, 0, 0, 0.5)
                glPushMatrix()
                draw_marker(verts, color)
                glPopMatrix()


            # Draw camera trace
            if len(self.cam_trace) > 0:
                draw_camera_trace(self.cam_trace)

            # Draw the camera frustum and origin using the 3d tranformation obtained from solvepnp
            if not self.cam_rot is None and not self.cam_trans is None:
                cam_trans = - self.cam_trans
                cam_rot = self.cam_rot.T
                rot_hm = np.eye(4, dtype=np.float32)
                rot_hm[:-1, :-1] = cam_rot
                trans_hm = np.eye(4, dtype=np.float32)
                trans_hm[:-1, -1] = cam_trans.reshape(3)
                cam_pose = np.matrix(rot_hm) * np.matrix(trans_hm)
                glPushMatrix()
                glMultMatrixf(cam_pose.T.flatten())
                # glMultMatrixf(self.cam_pose.flatten())
                draw_frustum(img_size, K, 150)
                glLineWidth(1)
                draw_frustum(img_size, K, .1)
                draw_coordinate_system(l=5)
                glPopMatrix()

            self.trackball.pop()

            glfwSwapBuffers(self._window)
            glfwMakeContextCurrent(active_window)

    def open_window(self):
        if not self._window:
            if self.fullscreen:
                monitor = glfwGetMonitors()[self.monitor_idx]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width= 640,int(640./(self.real_world_size['x']/self.real_world_size['y'])) #open with same aspect ratio as surface

            self._window = glfwCreateWindow(height, width, self.name, monitor=monitor, share=glfwGetCurrentContext())
            if not self.fullscreen:
                glfwSetWindowPos(self._window,self.window_position_default[0],self.window_position_default[1])



            self.trackball = Trackball()
            self.input = {'down':False, 'mouse':(0,0)}


            #Register callbacks
            glfwSetFramebufferSizeCallback(self._window,self.on_resize)
            glfwSetKeyCallback(self._window,self.on_window_key)
            glfwSetWindowCloseCallback(self._window,self.on_close)
            glfwSetMouseButtonCallback(self._window,self.on_window_mouse_button)
            glfwSetCursorPosCallback(self._window,self.on_pos)
            glfwSetScrollCallback(self._window,self.on_scroll)

            self.on_resize(self._window,*glfwGetFramebufferSize(self._window))

            # gl_state settings
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()
            make_coord_system_norm_based()

            # refresh speed settings
            glfwSwapInterval(0)

            glfwMakeContextCurrent(active_window)

    def close_window(self):
        if self._window:
            glfwDestroyWindow(self._window)
            self._window = None
            self.window_should_close = False

    def open_close_window(self):
        if self._window:
            self.close_window()
        else:
            self.open_window()


    # window calbacks
    def on_resize(self,window,w, h):
        self.trackball.set_window_size(w,h)
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(window)
        adjust_gl_view(w,h)
        glfwMakeContextCurrent(active_window)

    def on_window_key(self,window, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                self.on_close()

    def on_close(self,window=None):
        self.close_window()


    def on_window_mouse_button(self,window,button, action, mods):
        if action == GLFW_PRESS:
            self.input['down'] = True
            self.input['mouse'] = glfwGetCursorPos(window)
        if action == GLFW_RELEASE:
            self.input['down'] = False


    def on_pos(self,window,x, y):
        if self.input['down']:
            old_x,old_y = self.input['mouse']
            self.trackball.drag_to(x-old_x,y-old_y)
            self.input['mouse'] = x,y


    def on_scroll(self,window,x,y):
        self.trackball.zoom_to(y)

    def cleanup(self):
        # """ called when the plugin gets terminated.
        # This happens either voluntarily or forced.
        # if you have a GUI or glfw window destroy it here.
        # """
        # self.save_surface_definitions_to_file()
        #
        # for s in self.surfaces:
        #     s.cleanup()
        # self.deinit_gui()
        pass

def draw_marker(verts, color):
    glColor4f(*color)
    glBegin(GL_LINE_LOOP)
    glVertex3f(*verts[0])
    glVertex3f(*verts[1])
    glVertex3f(*verts[1])
    glVertex3f(*verts[2])
    glVertex3f(*verts[2])
    glVertex3f(*verts[3])
    glVertex3f(*verts[3])
    glVertex3f(*verts[0])
    glEnd()

def draw_camera_trace(trace):
    glColor4f(0,0,1,0.5)
    for i in range(len(trace)-1):
        glBegin(GL_LINES)
        glVertex3f(*trace[i])
        glVertex3f(*trace[i+1])
        glEnd()


def draw_frustum(img_size, K, scale=1):
    # average focal length
    f = (K[0, 0] + K[1, 1]) / 2
    # compute distances for setting up the camera pyramid
    W = 0.5*(img_size[0])
    H = 0.5*(img_size[1])
    Z = f
    # scale the pyramid
    W /= scale
    H /= scale
    Z /= scale
    # draw it
    glColor4f( 1, 0.5, 0, 0.5 )
    glBegin( GL_LINE_LOOP )
    glVertex3f( 0, 0, 0 )
    glVertex3f( -W, H, Z )
    glVertex3f( W, H, Z )
    glVertex3f( 0, 0, 0 )
    glVertex3f( W, H, Z )
    glVertex3f( W, -H, Z )
    glVertex3f( 0, 0, 0 )
    glVertex3f( W, -H, Z )
    glVertex3f( -W, -H, Z )
    glVertex3f( 0, 0, 0 )
    glVertex3f( -W, -H, Z )
    glVertex3f( -W, H, Z )
    glEnd( )

def draw_coordinate_system(l=1):
    # Draw x-axis line.
    glColor3f(1, 0, 0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(l, 0, 0)
    glEnd()

    # Draw y-axis line.
    glColor3f(0, 1, 0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, l, 0)
    glEnd()

    # Draw z-axis line.
    glColor3f(0, 0, 1)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, l)
    glEnd()