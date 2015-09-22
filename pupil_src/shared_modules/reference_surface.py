'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import numpy as np
import cv2
from gl_utils import adjust_gl_view,clear_gl_screen,basic_gl_setup,cvmat_to_glmat,make_coord_system_norm_based
from gl_utils.trackball import Trackball
from glfw import *
from OpenGL.GL import *

from pyglui.cygl.utils import RGBA
from pyglui.cygl.utils import draw_polyline_norm,draw_polyline,draw_points_norm,draw_points
from OpenGL.GL import GL_LINES
from methods import GetAnglesPolyline,normalize

from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
#ctypes import for atb_vars:
from time import time

import logging
logger = logging.getLogger(__name__)

def m_verts_to_screen(verts):
    #verts need to be sorted counter-clockwise stating at bottom left
    mapped_space_one = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
    return cv2.getPerspectiveTransform(mapped_space_one,verts)

def m_verts_from_screen(verts):
    #verts need to be sorted counter-clockwise stating at bottom left
    mapped_space_one = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
    return cv2.getPerspectiveTransform(verts,mapped_space_one)



class Reference_Surface(object):
    """docstring for Reference Surface

    The surface coodinate system is 0-1.
    Origin is the bottom left corner, (1,1) is the top right

    The first scalar in the pos vector is width we call this 'u'.
    The second is height we call this 'v'.
    The surface is thus defined by 4 vertecies:
        Our convention is this order: (0,0),(1,0),(1,1),(0,1)

    The surface is supported by a set of n>=1 Markers:
        Each marker has an id, you can not not have markers with the same id twice.
        Each marker has 4 verts (order is the same as the surface verts)
        Each maker vertex has a uv coord that places it on the surface

    When we find the surface in locate() we use the correspondence
    of uv and screen coords of all 4 verts of all detected markers to get the
    surface to screen homography.

    This allows us to get homographies for partially visible surfaces,
    all we need are 2 visible markers. (We could get away with just
    one marker but in pracise this is to noisy.)
    The more markers we find the more accurate the homography.

    """
    def __init__(self,name="unnamed",saved_definition=None):
        self.name = name
        self.markers = {}
        self.detected_markers = 0
        self.defined = False
        self.build_up_status = 0
        self.required_build_up = 90.
        self.detected = False
        self.m_to_screen = None
        self.m_from_screen = None
        self.camera_pose_3d = None

        self.uid = str(time())
        self.real_world_size = {'x':1.,'y':1.}

        ###window and gui vars
        self._window = None
        self.fullscreen = False
        self.window_should_open = False
        self.window_should_close = False

        self.gaze_on_srf = [] # points on surface for realtime feedback display

        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans',get_opensans_font_path())
        self.glfont.set_size(22)
        self.glfont.set_color_float((0.2,0.5,0.9,1.0))


        if saved_definition is not None:
            self.load_from_dict(saved_definition)



    def save_to_dict(self):
        """
        save all markers and name of this surface to a dict.
        """
        markers = dict([(m_id,m.uv_coords) for m_id,m in self.markers.iteritems()])
        return {'name':self.name,'uid':self.uid,'markers':markers,'real_world_size':self.real_world_size}


    def load_from_dict(self,d):
        """
        load all markers of this surface to a dict.
        """
        self.name = d['name']
        self.uid = d['uid']
        self.real_world_size = d.get('real_world_size',{'x':1.,'y':1.})

        marker_dict = d['markers']
        for m_id,uv_coords in marker_dict.iteritems():
            self.markers[m_id] = Support_Marker(m_id)
            self.markers[m_id].load_uv_coords(uv_coords)

        #flag this surface as fully defined
        self.defined = True
        self.build_up_status = self.required_build_up

    def build_correspondance(self, visible_markers):
        """
        - use all visible markers
        - fit a convex quadrangle around it
        - use quadrangle verts to establish perpective transform
        - map all markers into surface space
        - build up list of found markers and their uv coords
        """
        if visible_markers == []:
            self.m_to_screen = None
            self.m_from_screen = None
            self.detected = False

            return

        all_verts = np.array([[m['verts_norm'] for m in visible_markers]])
        all_verts.shape = (-1,1,2) # [vert,vert,vert,vert,vert...] with vert = [[r,c]]
        hull = cv2.convexHull(all_verts,clockwise=False)

        #simplify until we have excatly 4 verts
        if hull.shape[0]>4:
            new_hull = cv2.approxPolyDP(hull,epsilon=1,closed=True)
            if new_hull.shape[0]>=4:
                hull = new_hull
        if hull.shape[0]>4:
            curvature = abs(GetAnglesPolyline(hull,closed=True))
            most_acute_4_threshold = sorted(curvature)[3]
            hull = hull[curvature<=most_acute_4_threshold]

        #now we need to roll the hull verts until we have the right orientation:
        distance_to_origin = np.sqrt(hull[:,:,0]**2+hull[:,:,1]**2)
        top_left_idx = np.argmin(distance_to_origin)
        hull = np.roll(hull,-top_left_idx,axis=0)


        #based on these 4 verts we calculate the transformations into a 0,0 1,1 square space
        self.m_to_screen = m_verts_to_screen(hull)
        self.m_from_screen = m_verts_from_screen(hull)
        self.detected = True
        # map the markers vertices in to the surface space (one can think of these as texture coordinates u,v)
        marker_uv_coords =  cv2.perspectiveTransform(all_verts,self.m_from_screen)
        marker_uv_coords.shape = (-1,4,1,2) #[marker,marker...] marker = [ [[r,c]],[[r,c]] ]

        # build up a dict of discovered markers. Each with a history of uv coordinates
        for m,uv in zip (visible_markers,marker_uv_coords):
            try:
                self.markers[m['id']].add_uv_coords(uv)
            except KeyError:
                self.markers[m['id']] = Support_Marker(m['id'])
                self.markers[m['id']].add_uv_coords(uv)

        #average collection of uv correspondences accros detected markers
        self.build_up_status = sum([len(m.collected_uv_coords) for m in self.markers.values()])/float(len(self.markers))

        if self.build_up_status >= self.required_build_up:
            self.finalize_correnspondance()
            self.defined = True

    def finalize_correnspondance(self):
        """
        - prune markers that have been visible in less than x percent of frames.
        - of those markers select a good subset of uv coords and compute mean.
        - this mean value will be used from now on to estable surface transform
        """
        persistent_markers = {}
        for k,m in self.markers.iteritems():
            if len(m.collected_uv_coords)>self.required_build_up*.5:
                persistent_markers[k] = m
        self.markers = persistent_markers
        for m in self.markers.values():
            m.compute_robust_mean()


    def locate(self, visible_markers, locate_3d=False, camera_intrinsics = None):
        """
        - find overlapping set of surface markers and visible_markers
        - compute homography (and inverse) based on this subset
        """

        if not self.defined:
            self.build_correspondance(visible_markers)
        else:
            marker_by_id = dict([(m['id'],m) for m in visible_markers])
            visible_ids = set(marker_by_id.keys())
            requested_ids = set(self.markers.keys())
            overlap = visible_ids & requested_ids
            self.detected_markers = len(overlap)
            if len(overlap)>=min(1,len(requested_ids)):
                self.detected = True
                yx = np.array( [marker_by_id[i]['verts_norm'] for i in overlap] )
                uv = np.array( [self.markers[i].uv_coords for i in overlap] )
                yx.shape=(-1,1,2)
                uv.shape=(-1,1,2)
                # print 'uv',uv
                # print 'yx',yx
                self.m_to_screen,mask = cv2.findHomography(uv,yx)
                self.m_from_screen = np.linalg.inv(self.m_to_screen)
                #self.m_from_screen,mask = cv2.findHomography(yx,uv)

                if locate_3d:

                    K,dist_coef,img_size = camera_intrinsics

                    ###marker support pose estiamtion:
                    # denormalize image reference points to pixel space
                    yx.shape = -1,2
                    yx *= img_size
                    yx.shape = -1, 1, 2
                    # scale normalized object points to world space units (think m,cm,mm)
                    uv.shape = -1,2
                    uv *= [self.real_world_size['x'], self.real_world_size['y']]
                    # convert object points to lie on z==0 plane in 3d space
                    uv3d = np.zeros((uv.shape[0], uv.shape[1]+1))
                    uv3d[:,:-1] = uv
                    # compute pose of object relative to camera center
                    self.is3dPoseAvailable, rot3d_cam_to_object, translate3d_cam_to_object = cv2.solvePnP(uv3d, yx, K, dist_coef,flags=cv2.CV_EPNP)

                    # not verifed, potentially usefull info: http://stackoverflow.com/questions/17423302/opencv-solvepnp-tvec-units-and-axes-directions

                    ###marker posed estimation from virtually projected points.
                    # object_pts = np.array([[[0,0],[0,1],[1,1],[1,0]]],dtype=np.float32)
                    # projected_pts = cv2.perspectiveTransform(object_pts,self.m_to_screen)
                    # projected_pts.shape = -1,2
                    # projected_pts *= img_size
                    # projected_pts.shape = -1, 1, 2
                    # # scale object points to world space units (think m,cm,mm)
                    # object_pts.shape = -1,2
                    # object_pts *= self.real_world_size
                    # # convert object points to lie on z==0 plane in 3d space
                    # object_pts_3d = np.zeros((4,3))
                    # object_pts_3d[:,:-1] = object_pts
                    # self.is3dPoseAvailable, rot3d_cam_to_object, translate3d_cam_to_object = cv2.solvePnP(object_pts_3d, projected_pts, K, dist_coef,flags=cv2.CV_EPNP)


                    # transformation from Camera Optical Center:
                    #   first: translate from Camera center to object origin.
                    #   second: rotate x,y,z
                    #   coordinate system is x,y,z where z goes out from the camera into the viewed volume.
                    # print rot3d_cam_to_object[0],rot3d_cam_to_object[1],rot3d_cam_to_object[2], translate3d_cam_to_object[0],translate3d_cam_to_object[1],translate3d_cam_to_object[2]

                    #turn translation vectors into 3x3 rot mat.
                    rot3d_cam_to_object_mat, _ = cv2.Rodrigues(rot3d_cam_to_object)

                    #to get the transformation from object to camera we need to reverse rotation and translation
                    translate3d_object_to_cam = - translate3d_cam_to_object
                    # rotation matrix inverse == transpose
                    rot3d_object_to_cam_mat = rot3d_cam_to_object_mat.T


                    # we assume that the volume of the object grows out of the marker surface and not into it. We thus have to flip the z-Axis:
                    flip_z_axix_hm = np.eye(4, dtype=np.float32)
                    flip_z_axix_hm[2,2] = -1
                    # create a homogenous tranformation matrix from the rotation mat
                    rot3d_object_to_cam_hm = np.eye(4, dtype=np.float32)
                    rot3d_object_to_cam_hm[:-1,:-1] = rot3d_object_to_cam_mat
                    # create a homogenous tranformation matrix from the translation vect
                    translate3d_object_to_cam_hm = np.eye(4, dtype=np.float32)
                    translate3d_object_to_cam_hm[:-1, -1] = translate3d_object_to_cam.reshape(3)

                    # combine all tranformations into transformation matrix that decribes the move from object origin and orientation to camera origin and orientation
                    tranform3d_object_to_cam =  np.matrix(flip_z_axix_hm) * np.matrix(rot3d_object_to_cam_hm) * np.matrix(translate3d_object_to_cam_hm)
                    self.camera_pose_3d = tranform3d_object_to_cam
                else:
                    self.is3dPoseAvailable = False

            else:
                self.detected = False
                self.is3dPoseAvailable = False

                self.m_from_screen = None
                self.m_to_screen = None


    def img_to_ref_surface(self,pos):
        if self.m_from_screen is not None:
            #convenience lines to allow 'simple' vectors (x,y) to be used
            shape = pos.shape
            pos.shape = (-1,1,2)
            new_pos = cv2.perspectiveTransform(pos,self.m_from_screen )
            new_pos.shape = shape
            return new_pos
        else:
            return None

    def ref_surface_to_img(self,pos):
        if self.m_to_screen is not None:
            #convenience lines to allow 'simple' vectors (x,y) to be used
            shape = pos.shape
            pos.shape = (-1,1,2)
            new_pos = cv2.perspectiveTransform(pos,self.m_to_screen )
            new_pos.shape = shape
            return new_pos
        else:
            return None



    def move_vertex(self,vert_idx,new_pos):
        """
        this fn is used to manipulate the surface boundary (coordinate system)
        new_pos is in uv-space coords
        if we move one vertex of the surface we need to find
        the tranformation from old quadrangle to new quardangle
        and apply that transformation to our marker uv-coords
        """
        before = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
        after = before.copy()
        after[vert_idx] = new_pos
        transform = cv2.getPerspectiveTransform(after,before)
        for m in self.markers.values():
            m.uv_coords = cv2.perspectiveTransform(m.uv_coords,transform)


    def marker_status(self):
        return "%s   %s/%s" %(self.name,self.detected_markers,len(self.markers))



    def gl_draw_frame(self,img_size):
        """
        draw surface and markers
        """
        if self.detected:
            frame = np.array([[[0,0],[1,0],[1,1],[0,1],[0,0]]],dtype=np.float32)
            hat = np.array([[[.3,.7],[.7,.7],[.5,.9],[.3,.7]]],dtype=np.float32)
            hat = cv2.perspectiveTransform(hat,self.m_to_screen)
            frame = cv2.perspectiveTransform(frame,self.m_to_screen)
            alpha = min(1,self.build_up_status/self.required_build_up)
            draw_polyline_norm(frame.reshape((5,2)),1,RGBA(1.0,0.2,0.6,alpha))
            draw_polyline_norm(hat.reshape((4,2)),1,RGBA(1.0,0.2,0.6,alpha))

            draw_points_norm(frame.reshape((5,-1))[0:1])
            text_anchor = frame.reshape((5,-1))[2]
            text_anchor[1] = 1-text_anchor[1]
            text_anchor *=img_size[1],img_size[0]
            self.glfont.draw_text(text_anchor[0],text_anchor[1],self.marker_status())

    def gl_draw_corners(self):
        """
        draw surface and markers
        """
        if self.detected:
            frame = np.array([[[0,0],[1,0],[1,1],[0,1]]],dtype=np.float32)
            frame = cv2.perspectiveTransform(frame,self.m_to_screen)
            draw_points_norm(frame.reshape((4,2)),15,RGBA(1.0,0.2,0.6,.5))



    #### fns to draw surface in separate window
    def gl_display_in_window(self,world_tex):
        """
        here we map a selected surface onto a seperate window.
        """
        if self._window and self.detected:
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            clear_gl_screen()

            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_from_screen)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, 1, 0, 1,-1,1) # gl coord convention

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            #apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            world_tex.draw()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

            # now lets get recent pupil positions on this surface:
            draw_points_norm(self.gaze_on_srf,color=RGBA(0.,8.,.5,.8), size=80)

            glfwSwapBuffers(self._window)
            glfwMakeContextCurrent(active_window)
        if self.window_should_close:
            self.close_window()

    #### fns to draw surface in separate window
    def gl_display_in_window_3d(self,world_tex,camera_intrinsics):
        """
        here we map a selected surface onto a seperate window.
        """
        K,dist_coef,img_size = camera_intrinsics

        if self._window and self.detected:
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
            glPushMatrix()
            glScalef(self.real_world_size['x'],self.real_world_size['y'],1)
            draw_polyline([[0,0],[0,1],[1,1],[1,0]],color = RGBA(.5,.3,.1,.5),thickness=3)
            glPopMatrix()
            # Draw the world window as projected onto the plane using the homography mapping
            glPushMatrix()
            glScalef(self.real_world_size['x'], self.real_world_size['y'], 1)
            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_from_screen)
            glMultMatrixf(m)
            glTranslatef(0,0,-.01)
            world_tex.draw()
            draw_polyline([[0,0],[0,1],[1,1],[1,0]],color = RGBA(.5,.3,.6,.5),thickness=3)
            glPopMatrix()

            # Draw the camera frustum and origin using the 3d tranformation obtained from solvepnp
            glPushMatrix()
            glMultMatrixf(self.camera_pose_3d.T.flatten())
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

            self._window = glfwCreateWindow(height, width, "Reference Surface: " + self.name, monitor=monitor, share=glfwGetCurrentContext())
            if not self.fullscreen:
                glfwSetWindowPos(self._window,200,0)



            self.trackball = Trackball()
            self.input = {'down':False, 'mouse':(0,0)}


            #Register callbacks
            glfwSetFramebufferSizeCallback(self._window,self.on_resize)
            glfwSetKeyCallback(self._window,self.on_key)
            glfwSetWindowCloseCallback(self._window,self.on_close)
            glfwSetMouseButtonCallback(self._window,self.on_button)
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

    def on_key(self,window, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                self.on_close()

    def on_close(self,window=None):
        self.window_should_close = True


    def on_button(self,window,button, action, mods):
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
        if self._window:
            self.close_window()




class Support_Marker(object):
    '''
    This is a class only to be used by Reference_Surface
    it decribes the used markers with the uv coords of its verts.
    '''
    def __init__(self,uid):
        self.uid = uid
        self.uv_coords = None
        self.collected_uv_coords = []

    def load_uv_coords(self,uv_coords):
        self.uv_coords = uv_coords

    def add_uv_coords(self,uv_coords):
        self.collected_uv_coords.append(uv_coords)

    def compute_robust_mean(self,threshhold=.1):
        """
        right now its just the mean. Not so good...
        """
        # a stacked list of marker uv coords. marker uv cords are 4 verts with each a uv position.
        uv = np.array(self.collected_uv_coords)
        # # the mean marker uv_coords including outliers
        # base_line_mean = np.mean(uv,axis=0)
        # # devidation is the distance of each scalar (4*2 per marker to the mean value of this scalar acros our stacked list)
        # deviation = uv-base_line_mean
        # # now we treat the four uv scalars as a vector in 8-d space and compute the euclidian distace to the mean
        # distance =  np.linalg.norm(deviation,axis=(1,3))
        # # we now have 1 distance measure per recorded apprearace of the marker
        # uv_subset = uv[distance<threshhold]
        # ratio = uv_subset.shape[0]/float(uv.shape[0])
        # #todo: find a good way to get some meaningfull and accurate numbers to use
        #right now we take the mean of the last 30 datapoints
        uv_mean = np.mean(uv[-30:],axis=0)
        self.uv_coords = uv_mean


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
    glColor3f( 1, 0, 0 )
    glBegin( GL_LINES )
    glVertex3f( 0, 0, 0 )
    glVertex3f( l, 0, 0 )
    glEnd( )

    # Draw y-axis line.
    glColor3f( 0, 1, 0 )
    glBegin( GL_LINES )
    glVertex3f( 0, 0, 0 )
    glVertex3f( 0, l, 0 )
    glEnd( )

    # Draw z-axis line.
    glColor3f( 0, 0, 1 )
    glBegin( GL_LINES )
    glVertex3f( 0, 0, 0 )
    glVertex3f( 0, 0, l )
    glEnd( )


if __name__ == '__main__':

    rotation3d = np.array([1,2,3],dtype=np.float32)
    translation3d = np.array([50,60,70],dtype=np.float32)

    # transformation from Camera Optical Center:
    #   first: translate from Camera center to object origin.
    #   second: rotate x,y,z
    #   coordinate system is x,y,z positive (not like opengl, where the z-axis is flipped.)
    # print rotation3d[0],rotation3d[1],rotation3d[2], translation3d[0],translation3d[1],translation3d[2]

    #turn translation vectors into 3x3 rot mat.
    rotation3dMat, _ = cv2.Rodrigues(rotation3d)


    #to get the transformation from object to camera we need to reverse rotation and translation
    #
    tranform3d_to_camera_translation = np.eye(4, dtype=np.float32)
    tranform3d_to_camera_translation[:-1, -1] = - translation3d

    #rotation matrix inverse == transpose
    tranform3d_to_camera_rotation = np.eye(4, dtype=np.float32)
    tranform3d_to_camera_rotation[:-1,:-1] = rotation3dMat.T

    print tranform3d_to_camera_translation
    print tranform3d_to_camera_rotation
    print np.matrix(tranform3d_to_camera_rotation) * np.matrix(tranform3d_to_camera_translation)




    rMat, _ = cv2.Rodrigues(rotation3d)
    self.from_camera_to_referece = np.eye(4, dtype=np.float32)
    self.from_camera_to_referece[:-1,:-1] = rMat
    self.from_camera_to_referece[:-1, -1] = translation3d.reshape(3)
    # self.camera_pose_3d = np.linalg.inv(self.camera_pose_3d)
