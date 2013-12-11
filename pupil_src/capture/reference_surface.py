'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import numpy as np
import cv2
from gl_utils import draw_gl_polyline,draw_gl_point,draw_gl_point_norm,draw_gl_points
from methods import GetAnglesPolyline,normalize

#ctypes import for atb_vars:
from ctypes import create_string_buffer


def m_verts_to_screen(verts):
    #verts need to be sorted counterclockwise stating at bottom left
    mapped_space_one = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
    return cv2.getPerspectiveTransform(mapped_space_one,verts)

def m_verts_from_screen(verts):
    #verts need to be sorted counterclockwise stating at bottom left
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

        if saved_definition is not None:
            self.load_from_dict(saved_definition)



    def save_to_dict(self):
        """
        save all markers and name of this surface to a dict.
        """
        markers = dict([(m_id,m.uv_coords) for m_id,m in self.markers.iteritems()])
        return {'name':self.name,'markers':markers}


    def load_from_dict(self,d):
        """
        load all markers of this surface to a dict.
        """
        self.name = d['name']
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
            return

        all_verts = np.array([[m['verts'] for m in visible_markers]])
        all_verts.shape = (-1,1,2) # [vert,vert,vert,vert,vert...] with vert = [[r,c]]
        hull = cv2.convexHull(all_verts,clockwise=True)

        #simplyfy until we have excatly 4 verts
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
        hull = np.roll(hull,3-top_left_idx,axis=0)


        #based on these 4 verts we calculate the transformations into a 0,0 1,1 square space
        self.m_to_screen = m_verts_to_screen(hull)
        self.m_from_screen = m_verts_from_screen(hull)

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


    def locate(self, visible_markers):
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
            if len(overlap)>=min(2,len(requested_ids)):
                self.detected = True
                yx = np.array( [marker_by_id[i]['verts'] for i in overlap] )
                uv = np.array( [self.markers[i].uv_coords for i in overlap] )
                yx.shape=(-1,1,2)
                uv.shape=(-1,1,2)
                # print 'uv',uv
                # print 'yx',yx
                self.m_to_screen,mask = cv2.findHomography(uv,yx)
                self.m_from_screen,mask = cv2.findHomography(yx,uv)

            else:
                self.detected = False
                self.m_from_screen = None
                self.m_to_screen = None




    def xy_to_uv(self,pos):
        if self.m_from_screen is not None:
            #convenience lines to allow 'simple' vectors (x,y) to be used
            shape = pos.shape
            pos.shape = (-1,1,2)
            new_pos = cv2.perspectiveTransform(pos,self.m_from_screen )
            new_pos.shape = shape
            return new_pos
        else:
            return None

    def uv_to_xy(self,pos):
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

    def atb_marker_status(self):
        return create_string_buffer("%s / %s" %(self.detected_markers,len(self.markers)),512)

    def atb_get_name(self):
        return create_string_buffer(self.name,512)

    def atb_set_name(self,name):
        self.name = name.value

    def gl_draw_frame(self):
        """
        draw surface and markers
        """
        if self.m_to_screen is not None:
            frame = np.array([[[0,0],[0,1],[1,1],[1,0],[0,0]]],dtype=np.float32)
            hat = np.array([[[.3,.7],[.5,.9],[.7,.7],[.3,.7]]],dtype=np.float32)
            hat = cv2.perspectiveTransform(hat,self.m_to_screen)
            frame = cv2.perspectiveTransform(frame,self.m_to_screen)
            alpha = min(1,self.build_up_status/self.required_build_up)
            draw_gl_polyline(frame.reshape((5,2)),(1.0,0.2,0.6,alpha))
            draw_gl_polyline(hat.reshape((4,2)),(1.0,0.2,0.6,alpha))


    def gl_draw_corners(self):
        """
        draw surface and markers
        """
        if self.m_to_screen is not None:
            frame = np.array([[[0,0],[0,1],[1,1],[1,0]]],dtype=np.float32)
            frame = cv2.perspectiveTransform(frame,self.m_to_screen)
            draw_gl_points(frame.reshape((4,2)),15,(1.0,0.2,0.6,.5))


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



