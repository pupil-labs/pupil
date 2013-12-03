import numpy as np
import cv2
from gl_utils import draw_gl_polyline,draw_gl_point,draw_gl_point_norm
from methods import GetAnglesPolyline

def m_verts_to_screen(verts):
    #verts need to be sorted counterclockwise stating at bottom left
    mapped_space_one = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
    return cv2.getPerspectiveTransform(mapped_space_one,verts)



class Reference_Surface(object):
    """docstring for Reference Surface"""
    def __init__(self,name="unnamed"):
        self.name = name
        self.markers = {}

        self.m_to_screen = None



    def build_correspondance(self, visible_markers):
        """
        - use all visible markers
        - fit a convex quadrangle around it
        - use quadrangle verts to establish perpective transform
        - map all markers into surface space
        - build up list of found markers and their uv coords
        """

        all_verts = np.array([[m['verts'] for m in visible_markers]])
        all_verts.shape = (-1,1,2) # [vert,vert,vert,vert,vert...] with vert = [[r,c]]
        hull = cv2.convexHull(all_verts,clockwise=True)

        #simplyfy until we have excaply 4 verts
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

        self.m_to_screen = m_verts_to_screen(hull)



        # for m in visible_markers:
        #     try:
        #         self.markers[m['id']].add_uv_coords()
        #     except KeyError:
        #         self.markers.append(Support_Marker[m['id']])
        #         self.markers[m['id']].add_uv_coords()

    def finalize_correnspondance(self):
        """
        - prune markers that have been visible in less than x percent of frames.
        - of those markers select a good subset of uv coords and compute mean.
        - this mean will be used from now on to estable surface transform
        """

    def locate(self, visible_markers):
        """
        - find overlapping set of surface markers and visible_markers
        - compute perspective transform (and inverse) based on this subset
        """

    def get_screen_to_surface_transform(self):
        """
        if ref surface was found return transformation to it
        """

    def get_surface_to_screen_transform(self):
        """
        if ref surface was found return transformation from it
        """

    def gl_draw(self):
        """
        draw surface and markers
        """
        if self.m_to_screen is not None:
            hat = np.array([[[0,0],[0,1],[.5,1.5],[1,1],[1,0],[0,0]]],dtype=np.float32)
            hat = cv2.perspectiveTransform(hat,self.m_to_screen)
            draw_gl_polyline(hat.reshape((6,2)),(1.0,0.2,0.6,1.0))
            draw_gl_point(hat.reshape((6,2))[0],15,(1.0,0.2,0.6,1.0))


class Support_Marker(object):
    '''

    '''
    def __init__(self,uid):
        self.uid = uid
        self.uv_coords = None
        self.collected_uv_coords = []

    def add_uv_coords(self,uv_coords):
        self.collected_uv_coords.append(uv_coords)

    def compute_robust_mean(self,threshhold=.01):
        # a stacked list of marker uv coords. marker uv cords are 4 verts with each a uv position.
        uv = np.array(self.collected_uv_coords)
        # the mean marker uv_coords including outliers
        base_line_mean = np.mean(uv,axis=0)
        # devidation is the distance of each scalar (4*2 per marker to the mean value of this scalar acros our stacked list)
        deviation = uv-base_line_mean
        # now we treat the four uv scalars as a vector in 8-d space and compute the euclidian distace to the mean
        distance =  np.linalg.norm(deviation,axis=(1,2))
        # we now have 1 distance measure per recorded apprearace of the marker
        uv_subset = uv[distance<threshhold]
        ratio = uv.shape[0]/float(uv_subset.shape[0])

        #todo: find a good way to get some meaningfull and accurate numbers to use
        uv_mean = np.mean(uv_subset,axis=0)
        self.uv_coords = uv_mean.tolist()



