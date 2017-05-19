'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import numpy as np
import cv2
from platform import system
from gl_utils import adjust_gl_view,clear_gl_screen,basic_gl_setup,cvmat_to_glmat,make_coord_system_norm_based
from gl_utils.trackball import Trackball
from glfw import *
from OpenGL.GL import *

from pyglui.cygl.utils import RGBA
from pyglui.cygl.utils import draw_polyline_norm,draw_polyline,draw_points_norm,draw_points
from OpenGL.GL import GL_LINES
from methods import GetAnglesPolyline,normalize,denormalize

from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
#ctypes import for atb_vars:
from time import time

import logging
logger = logging.getLogger(__name__)

marker_corners_norm = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
def m_verts_to_screen(verts):
    #verts need to be sorted counter-clockwise stating at bottom left
    return cv2.getPerspectiveTransform(marker_corners_norm,verts)

def m_verts_from_screen(verts):
    #verts need to be sorted counter-clockwise stating at bottom left
    return cv2.getPerspectiveTransform(verts,marker_corners_norm)



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
        self.use_distortion = 0

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


        self.old_corners_robust = None
        if saved_definition is not None:
            self.load_from_dict(saved_definition)

        # UI Platform tweaks
        if system() == 'Linux':
            self.window_position_default = (0, 0)
        elif system() == 'Windows':
            self.window_position_default = (8, 31)
        else:
            self.window_position_default = (0, 0)



    def save_to_dict(self):
        """
        save all markers and name of this surface to a dict.
        """
        markers = dict([(m_id,m.uv_coords.tolist()) for m_id,m in self.markers.items()])
        return {'name':self.name,'uid':self.uid,'markers':markers,'real_world_size':self.real_world_size}


    def load_from_dict(self,d):
        """
        load all markers of this surface to a dict.
        """
        self.name = d['name']
        self.uid = d['uid']
        self.real_world_size = d.get('real_world_size',{'x':1.,'y':1.})

        marker_dict = d['markers']
        for m_id,uv_coords in marker_dict.items():
            self.markers[m_id] = Support_Marker(m_id)
            self.markers[m_id].load_uv_coords(np.asarray(uv_coords))

        #flag this surface as fully defined
        self.defined = True
        self.build_up_status = self.required_build_up

    def build_correspondance(self, visible_markers,camera_calibration,min_marker_perimeter,min_id_confidence):
        """
        - use all visible markers
        - fit a convex quadrangle around it
        - use quadrangle verts to establish perpective transform
        - map all markers into surface space
        - build up list of found markers and their uv coords
        """
        usable_markers = [m for m in visible_markers if m['perimeter']>=min_marker_perimeter]
        all_verts = [m['verts'] for m in usable_markers]
        if not all_verts:
            return
        all_verts = np.array(all_verts,dtype=np.float32)
        all_verts.shape = (-1,1,2) # [vert,vert,vert,vert,vert...] with vert = [[r,c]]
        # all_verts_undistorted_normalized centered in img center flipped in y and range [-1,1]
        all_verts_undistorted_normalized = cv2.undistortPoints(all_verts, np.asarray(camera_calibration['camera_matrix']),np.asarray(camera_calibration['dist_coefs'])*self.use_distortion)
        hull = cv2.convexHull(all_verts_undistorted_normalized,clockwise=False)

        #simplify until we have excatly 4 verts
        if hull.shape[0]>4:
            new_hull = cv2.approxPolyDP(hull,epsilon=1,closed=True)
            if new_hull.shape[0]>=4:
                hull = new_hull
        if hull.shape[0]>4:
            curvature = abs(GetAnglesPolyline(hull,closed=True))
            most_acute_4_threshold = sorted(curvature)[3]
            hull = hull[curvature<=most_acute_4_threshold]


        # all_verts_undistorted_normalized space is flipped in y.
        # we need to change the order of the hull vertecies
        hull = hull[[1,0,3,2],:,:]

        # now we need to roll the hull verts until we have the right orientation:
        # all_verts_undistorted_normalized space has its origin at the image center.
        # adding 1 to the coordinates puts the origin at the top left.
        distance_to_top_left = np.sqrt((hull[:,:,0]+1)**2+(hull[:,:,1]+1)**2)
        bot_left_idx = np.argmin(distance_to_top_left)+1
        hull = np.roll(hull,-bot_left_idx,axis=0)

        #based on these 4 verts we calculate the transformations into a 0,0 1,1 square space
        m_from_undistored_norm_space = m_verts_from_screen(hull)
        self.detected = True
        # map the markers vertices into the surface space (one can think of these as texture coordinates u,v)
        marker_uv_coords =  cv2.perspectiveTransform(all_verts_undistorted_normalized,m_from_undistored_norm_space)
        marker_uv_coords.shape = (-1,4,1,2) #[marker,marker...] marker = [ [[r,c]],[[r,c]] ]

        # build up a dict of discovered markers. Each with a history of uv coordinates
        for m,uv in zip(usable_markers, marker_uv_coords):
            try:
                self.markers[m['id']].add_uv_coords(uv)
            except KeyError:
                self.markers[m['id']] = Support_Marker(m['id'])
                self.markers[m['id']].add_uv_coords(uv)

        #average collection of uv correspondences accros detected markers
        self.build_up_status = sum([len(m.collected_uv_coords) for m in self.markers.values()])/float(len(self.markers))

        if self.build_up_status >= self.required_build_up:
            self.finalize_correnspondance()

    def finalize_correnspondance(self):
        """
        - prune markers that have been visible in less than x percent of frames.
        - of those markers select a good subset of uv coords and compute mean.
        - this mean value will be used from now on to estable surface transform
        """
        persistent_markers = {}
        for k,m in self.markers.items():
            if len(m.collected_uv_coords)>self.required_build_up*.5:
                persistent_markers[k] = m
        self.markers = persistent_markers
        for m in self.markers.values():
            m.compute_robust_mean()

        self.defined = True
        if hasattr(self,'on_finish_define'):
            self.on_finish_define()
            del self.on_finish_define


    def locate(self, visible_markers,camera_calibration,min_marker_perimeter,min_id_confidence, locate_3d=False,):
        """
        - find overlapping set of surface markers and visible_markers
        - compute homography (and inverse) based on this subset
        """

        if not self.defined:
            self.build_correspondance(visible_markers,camera_calibration,min_marker_perimeter,min_id_confidence)

        res = self._get_location(visible_markers,camera_calibration,min_marker_perimeter,min_id_confidence,locate_3d)
        self.detected = res['detected']
        self.detected_markers = res['detected_markers']
        self.m_to_screen = res['m_to_screen']
        self.m_from_screen = res['m_from_screen']
        self.camera_pose_3d = res['camera_pose_3d']

    def _get_location(self,visible_markers,camera_calibration,min_marker_perimeter,min_id_confidence, locate_3d=False):

        filtered_markers = [m for m in visible_markers if m['perimeter']>=min_marker_perimeter and m['id_confidence']>min_id_confidence]
        marker_by_id ={}
        #if an id shows twice we use the bigger marker (usually this is a screen camera echo artifact.)
        for m in filtered_markers:
            if m["id"] in marker_by_id and m["perimeter"] < marker_by_id[m['id']]['perimeter']:
                pass
            else:
                marker_by_id[m["id"]] = m

        visible_ids = set(marker_by_id.keys())
        requested_ids = set(self.markers.keys())
        overlap = visible_ids & requested_ids
        # need at least two markers per surface when the surface is more that 1 marker.
        if overlap and len(overlap) >= min(2,len(requested_ids)):
            detected = True
            xy = np.array( [marker_by_id[i]['verts'] for i in overlap] )
            uv = np.array( [self.markers[i].uv_coords for i in overlap] )
            uv.shape=(-1,1,2)

            # our camera lens creates distortions we want to get a good 2d estimate despite that so we:
            # compute the homography transform from marker into the undistored normalized image space
            # (the line below is the same as what you find in methods.undistort_unproject_pts, except that we ommit the z corrd as it is always one.)
            xy_undistorted_normalized = cv2.undistortPoints(xy.reshape(-1,1,2), np.asarray(camera_calibration['camera_matrix']),np.asarray(camera_calibration['dist_coefs'])*self.use_distortion)
            m_to_undistored_norm_space,mask = cv2.findHomography(uv,xy_undistorted_normalized, method=cv2.RANSAC,ransacReprojThreshold=0.1)
            if not mask.all():
                detected = False
            m_from_undistored_norm_space,mask = cv2.findHomography(xy_undistorted_normalized,uv)
            # project the corners of the surface to undistored space
            corners_undistored_space = cv2.perspectiveTransform(marker_corners_norm.reshape(-1,1,2),m_to_undistored_norm_space)
            # project and distort these points  and normalize them
            corners_redistorted, corners_redistorted_jacobian = cv2.projectPoints(cv2.convertPointsToHomogeneous(corners_undistored_space), np.array([0,0,0], dtype=np.float32) , np.array([0,0,0], dtype=np.float32), np.asarray(camera_calibration['camera_matrix']), np.asarray(camera_calibration['dist_coefs'])*self.use_distortion)
            corners_nulldistorted, corners_nulldistorted_jacobian = cv2.projectPoints(cv2.convertPointsToHomogeneous(corners_undistored_space), np.array([0,0,0], dtype=np.float32) , np.array([0,0,0], dtype=np.float32), np.asarray(camera_calibration['camera_matrix']), np.asarray(camera_calibration['dist_coefs'])*0)

            #normalize to pupil norm space
            corners_redistorted.shape = -1,2
            corners_redistorted /= camera_calibration['resolution']
            corners_redistorted[:,-1] = 1-corners_redistorted[:,-1]

            #normalize to pupil norm space
            corners_nulldistorted.shape = -1,2
            corners_nulldistorted /= camera_calibration['resolution']
            corners_nulldistorted[:,-1] = 1-corners_nulldistorted[:,-1]


            # maps for extreme lens distortions will behave irratically beyond the image bounds
            # since our surfaces often extend beyond the screen we need to interpolate
            # between a distored projection and undistored one.

            # def ratio(val):
            #     centered_val = abs(.5 - val)
            #     # signed distance to img cennter .5 is imag bound
            #     # we look to interpolate between .7 and .9
            #     inter = max()

            corners_robust = []
            for nulldist,redist in zip(corners_nulldistorted,corners_redistorted):
                if -.4 < nulldist[0] <1.4 and -.4 < nulldist[1] <1.4:
                    corners_robust.append(redist)
                else:
                    corners_robust.append(nulldist)


            corners_robust = np.array(corners_robust)

            if self.old_corners_robust is not None and np.mean(np.abs(corners_robust-self.old_corners_robust)) < 0.02:
                smooth_corners_robust  = self.old_corners_robust
                smooth_corners_robust += .5*(corners_robust-self.old_corners_robust )

                corners_robust = smooth_corners_robust
                self.old_corners_robust  = smooth_corners_robust
            else:
                self.old_corners_robust = corners_robust

            #compute a perspective thransform from from the marker norm space to the apparent image.
            # The surface corners will be at the right points
            # However the space between the corners may be distored due to distortions of the lens,
            m_to_screen = m_verts_to_screen(corners_robust)
            m_from_screen = m_verts_from_screen(corners_robust)

            camera_pose_3d = None
            if locate_3d:
                dist_coef, = np.asarray(camera_calibration['dist_coefs'])
                img_size = camera_calibration['resolution']
                K = np.asarray(camera_calibration['camera_matrix'])

                # 3d marker support pose estimation:
                # scale normalized object points to world space units (think m,cm,mm)
                uv.shape = -1,2
                uv *= [self.real_world_size['x'], self.real_world_size['y']]
                # convert object points to lie on z==0 plane in 3d space
                uv3d = np.zeros((uv.shape[0], uv.shape[1]+1))
                uv3d[:,:-1] = uv
                xy.shape = -1,1,2
                # compute pose of object relative to camera center
                is3dPoseAvailable, rot3d_cam_to_object, translate3d_cam_to_object = cv2.solvePnP(uv3d, xy, K, dist_coef,flags=cv2.SOLVEPNP_EPNP)

                if is3dPoseAvailable:

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
                    camera_pose_3d = tranform3d_object_to_cam
            if detected == False:
                camera_pose_3d = None
                m_from_screen = None
                m_to_screen = None
                m_from_undistored_norm_space = None
                m_to_undistored_norm_space = None

        else:
            detected = False
            camera_pose_3d = None
            m_from_screen = None
            m_to_screen = None
            m_from_undistored_norm_space = None
            m_to_undistored_norm_space = None

        return {'detected':detected,'detected_markers':len(overlap),'m_from_undistored_norm_space':m_from_undistored_norm_space,'m_to_undistored_norm_space':m_to_undistored_norm_space,'m_from_screen':m_from_screen,'m_to_screen':m_to_screen,'camera_pose_3d':camera_pose_3d}


    def img_to_ref_surface(self,pos):
        #convenience lines to allow 'simple' vectors (x,y) to be used
        shape = pos.shape
        pos.shape = (-1,1,2)
        new_pos = cv2.perspectiveTransform(pos,self.m_from_screen )
        new_pos.shape = shape
        return new_pos


    def ref_surface_to_img(self,pos):
        #convenience lines to allow 'simple' vectors (x,y) to be used
        shape = pos.shape
        pos.shape = (-1,1,2)
        new_pos = cv2.perspectiveTransform(pos,self.m_to_screen )
        new_pos.shape = shape
        return new_pos


    @staticmethod
    def map_datum_to_surface(d,m_from_screen):
        pos = np.array([d['norm_pos']]).reshape(1,1,2)
        mapped_pos = cv2.perspectiveTransform(pos , m_from_screen )
        mapped_pos.shape = (2)
        on_srf = bool((0 <= mapped_pos[0] <= 1) and (0 <= mapped_pos[1] <= 1))
        return {'topic':d['topic']+"_on_surface",'norm_pos':(mapped_pos[0],mapped_pos[1]),'confidence':d['confidence'],'on_srf':on_srf,'base_data':d }

    def map_data_to_surface(self,data,m_from_screen):
        return [self.map_datum_to_surface(d,m_from_screen) for d in data]

    def move_vertex(self,vert_idx,new_pos):
        """
        this fn is used to manipulate the surface boundary (coordinate system)
        new_pos is in uv-space coords
        if we move one vertex of the surface we need to find
        the tranformation from old quadrangle to new quardangle
        and apply that transformation to our marker uv-coords
        """
        before = marker_corners_norm
        after = before.copy()
        after[vert_idx] = new_pos
        transform = cv2.getPerspectiveTransform(after,before)
        for m in self.markers.values():
            m.uv_coords = cv2.perspectiveTransform(m.uv_coords,transform)


    def add_marker(self,marker,visible_markers,camera_calibration,min_marker_perimeter,min_id_confidence):
        '''
        add marker to surface.
        '''
        res = self._get_location(visible_markers,camera_calibration,min_marker_perimeter,min_id_confidence,locate_3d=False)
        if res['detected']:
            support_marker = Support_Marker(marker['id'])
            marker_verts = np.array(marker['verts'])
            marker_verts.shape = (-1,1,2)
            marker_verts_undistorted_normalized = cv2.undistortPoints(marker_verts, np.asarray(camera_calibration['camera_matrix']),np.asarray(camera_calibration['dist_coefs'])*self.use_distortion)
            marker_uv_coords =  cv2.perspectiveTransform(marker_verts_undistorted_normalized,res['m_from_undistored_norm_space'])
            support_marker.load_uv_coords(marker_uv_coords)
            self.markers[marker['id']] = support_marker


    def remove_marker(self,marker):
        if len(self.markers) == 1:
            logger.warning("Need at least one marker per surface. Will not remove this last marker.")
            return
        self.markers.pop(marker['id'])

    def marker_status(self):
        return "{}   {}/{}".format(self.name, self.detected_markers, len(self.markers))

    def get_mode_toggle(self,pos,img_shape):
        if self.detected and self.defined:
            x,y = pos
            frame = np.array([[[0,0],[1,0],[1,1],[0,1],[0,0]]],dtype=np.float32)
            frame = cv2.perspectiveTransform(frame,self.m_to_screen)
            text_anchor = frame.reshape((5,-1))[2]
            text_anchor[1] = 1-text_anchor[1]
            text_anchor *=img_shape[1],img_shape[0]
            text_anchor = text_anchor[0],text_anchor[1]-75
            surface_edit_anchor = text_anchor[0],text_anchor[1]+25
            marker_edit_anchor = text_anchor[0],text_anchor[1]+50
            if np.sqrt((x-surface_edit_anchor[0])**2 + (y-surface_edit_anchor[1])**2) <15:
                return 'surface_mode'
            elif np.sqrt((x-marker_edit_anchor[0])**2 + (y-marker_edit_anchor[1])**2) <15:
                return 'marker_mode'
            else:
                return None
        else:
            return None

    def gl_draw_frame(self,img_size,color = (1.0,0.2,0.6,1.0),highlight=False,surface_mode=False,marker_mode=False):
        """
        draw surface and markers
        """
        if self.detected:
            r,g,b,a = color
            frame = np.array([[[0,0],[1,0],[1,1],[0,1],[0,0]]],dtype=np.float32)
            hat = np.array([[[.3,.7],[.7,.7],[.5,.9],[.3,.7]]],dtype=np.float32)
            hat = cv2.perspectiveTransform(hat,self.m_to_screen)
            frame = cv2.perspectiveTransform(frame,self.m_to_screen)
            alpha = min(1,self.build_up_status/self.required_build_up)
            if highlight:
                draw_polyline_norm(frame.reshape((5,2)),1,RGBA(r,g,b,a*.1),line_type=GL_POLYGON)
            draw_polyline_norm(frame.reshape((5,2)),1,RGBA(r,g,b,a*alpha))
            draw_polyline_norm(hat.reshape((4,2)),1,RGBA(r,g,b,a*alpha))
            text_anchor = frame.reshape((5,-1))[2]
            text_anchor[1] = 1-text_anchor[1]
            text_anchor *=img_size[1],img_size[0]
            text_anchor = text_anchor[0],text_anchor[1]-75
            surface_edit_anchor = text_anchor[0],text_anchor[1]+25
            marker_edit_anchor = text_anchor[0],text_anchor[1]+50
            if self.defined:
                if marker_mode:
                    draw_points([marker_edit_anchor],color=RGBA(0,.8,.7))
                else:
                    draw_points([marker_edit_anchor])
                if surface_mode:
                    draw_points([surface_edit_anchor],color=RGBA(0,.8,.7))
                else:
                    draw_points([surface_edit_anchor])

                self.glfont.set_blur(3.9)
                self.glfont.set_color_float((0,0,0,.8))
                self.glfont.draw_text(text_anchor[0]+15,text_anchor[1]+6,self.marker_status())
                self.glfont.draw_text(surface_edit_anchor[0]+15,surface_edit_anchor[1]+6,'edit surface')
                self.glfont.draw_text(marker_edit_anchor[0]+15,marker_edit_anchor[1]+6,'add/remove markers')
                self.glfont.set_blur(0.0)
                self.glfont.set_color_float((0.1,8.,8.,.9))
                self.glfont.draw_text(text_anchor[0]+15,text_anchor[1]+6,self.marker_status())
                self.glfont.draw_text(surface_edit_anchor[0]+15,surface_edit_anchor[1]+6,'edit surface')
                self.glfont.draw_text(marker_edit_anchor[0]+15,marker_edit_anchor[1]+6,'add/remove markers')
            else:
                progress = (self.build_up_status/float(self.required_build_up))*100
                progress_text = '%.0f%%'%progress
                self.glfont.set_blur(3.9)
                self.glfont.set_color_float((0,0,0,.8))
                self.glfont.draw_text(text_anchor[0]+15,text_anchor[1]+6,self.marker_status())
                self.glfont.draw_text(surface_edit_anchor[0]+15,surface_edit_anchor[1]+6,'Learning affiliated markers...')
                self.glfont.draw_text(marker_edit_anchor[0]+15,marker_edit_anchor[1]+6,progress_text)
                self.glfont.set_blur(0.0)
                self.glfont.set_color_float((0.1,8.,8.,.9))
                self.glfont.draw_text(text_anchor[0]+15,text_anchor[1]+6,self.marker_status())
                self.glfont.draw_text(surface_edit_anchor[0]+15,surface_edit_anchor[1]+6,'Learning affiliated markers...')
                self.glfont.draw_text(marker_edit_anchor[0]+15,marker_edit_anchor[1]+6,progress_text)

    def gl_draw_corners(self):
        """
        draw surface and markers
        """
        if self.detected:
            frame = cv2.perspectiveTransform(marker_corners_norm.reshape(-1,1,2),self.m_to_screen)
            draw_points_norm(frame.reshape((4,2)),20,RGBA(1.0,0.2,0.6,.5))



    #### fns to draw surface in seperate window
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
            for gp in self.gaze_on_srf:
                draw_points_norm([gp['norm_pos']],color=RGBA(0.0,0.8,0.5,0.8), size=80)

            glfwSwapBuffers(self._window)
            glfwMakeContextCurrent(active_window)

    #### fns to draw surface in separate window
    def gl_display_in_window_3d(self,world_tex,camera_intrinsics):
        """
        here we map a selected surface onto a seperate window.
        """
        K,dist_coef,img_size = camera_intrinsics['camera_matrix'],camera_intrinsics['dist_coefs'],camera_intrinsics['resolution']

        if self._window and self.camera_pose_3d is not None:
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
                glfwSetWindowPos(self._window,self.window_position_default[0],self.window_position_default[1])



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
        self.close_window()


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
        self.robust_uv_cords = False

    def load_uv_coords(self,uv_coords):
        self.uv_coords = uv_coords
        self.robust_uv_cords = True

    def add_uv_coords(self,uv_coords):
        self.collected_uv_coords.append(uv_coords)
        self.uv_coords = uv_coords

    def compute_robust_mean(self,threshhold=.1):
        """
        treat 50% as outliers. Assume majory is right.
        """
        # a stacked list of marker uv coords. marker uv cords are 4 verts with each a uv position.
        uv = np.array(self.collected_uv_coords)
        # # the mean marker uv_coords including outliers
        base_line_mean = np.mean(uv,axis=0)
        # # devidation is the distance of each scalar (4*2 per marker to the mean value of this scalar acros our stacked list)
        deviation = uv-base_line_mean
        # # now we treat the four uv scalars as a vector in 8-d space and compute the distace to the mean
        distance =  np.linalg.norm(deviation,axis=(1,3)).reshape(-1)
        # lets get the .5 cutof;
        cut_off = sorted(distance)[len(distance)//2]
        # filter the better half
        uv_subset = uv[distance<=cut_off]
        # claculate the mean of this subset
        uv_mean = np.mean(uv_subset,axis=0)
        # use it
        self.uv_coords = uv_mean
        self.robust_uv_cords = True


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

    print(tranform3d_to_camera_translation)
    print(tranform3d_to_camera_rotation)
    print(np.matrix(tranform3d_to_camera_rotation) * np.matrix(tranform3d_to_camera_translation))




    # rMat, _ = cv2.Rodrigues(rotation3d)
    # self.from_camera_to_referece = np.eye(4, dtype=np.float32)
    # self.from_camera_to_referece[:-1,:-1] = rMat
    # self.from_camera_to_referece[:-1, -1] = translation3d.reshape(3)
    # # self.camera_pose_3d = np.linalg.inv(self.camera_pose_3d)
