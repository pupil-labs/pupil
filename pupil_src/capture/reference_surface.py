import numpy as np
import cv2
from gl_utils import draw_gl_polyline,draw_gl_point,draw_gl_point_norm
from methods import GetAnglesPolyline

def m_verts_to_screen(verts):
    #verts need to be sorted counterclockwise stating at bottom left
    mapped_space_one = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
    return cv2.getPerspectiveTransform(mapped_space_one,verts)

def m_verts_from_screen(verts):
    #verts need to be sorted counterclockwise stating at bottom left
    mapped_space_one = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
    return cv2.getPerspectiveTransform(verts,mapped_space_one)



class Reference_Surface(object):
    """docstring for Reference Surface"""
    def __init__(self,name="unnamed"):
        self.name = name
        self.markers = {}

        self.defined = False
        self.build_up_status = 0
        self.required_build_up = 30.
        self.m_to_screen = None
        self.m_from_screen = None


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

        #avg collection of uv correspondences acros detected markers
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
        - compute perspective transform (and inverse) based on this subset
        """

        if not self.defined:
            self.build_correspondance(visible_markers)
        else:
            marker_by_id = dict([(m['id'],m) for m in visible_markers])
            visible_ids = set(marker_by_id.keys())
            requested_ids = set(self.markers.keys())
            overlap = visible_ids & requested_ids
            if len(overlap)>=min(2,len(requested_ids)):
                yx = np.array( [marker_by_id[i]['verts'] for i in overlap] )
                uv = np.array( [self.markers[i].uv_coords for i in overlap] )
                yx.shape=(-1,1,2)
                uv.shape=(-1,1,2)
                # print 'uv',uv
                # print 'yx',yx
                self.m_to_screen,mask = cv2.findHomography(uv,yx)
                # self.m_from_screen,mask = cv2.findHomography(yx,uv)

            else:
                self.m_from_screen = None
                self.m_to_screen = None

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
            frame = np.array([[[0,0],[0,1],[1,1],[1,0],[0,0]]],dtype=np.float32)
            hat = np.array([[[.3,.7],[.5,.9],[.7,.7],[.3,.7]]],dtype=np.float32)
            hat = cv2.perspectiveTransform(hat,self.m_to_screen)
            frame = cv2.perspectiveTransform(frame,self.m_to_screen)
            alpha = min(1,self.build_up_status/self.required_build_up)
            draw_gl_polyline(frame.reshape((5,2)),(1.0,0.2,0.6,alpha))
            draw_gl_polyline(hat.reshape((4,2)),(1.0,0.2,0.6,alpha))
            draw_gl_point(frame.reshape((5,2))[0],15,(1.0,0.2,0.6,alpha))


class Support_Marker(object):
    '''

    '''
    def __init__(self,uid):
        self.uid = uid
        self.uv_coords = None
        self.collected_uv_coords = []

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
        uv_mean = np.mean(uv,axis=0)
        self.uv_coords = uv_mean



