'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import numpy as np
import cv2
from gl_utils import draw_gl_polyline,adjust_gl_view,draw_gl_polyline_norm,clear_gl_screen,draw_gl_point,draw_gl_points,draw_gl_point_norm,draw_gl_points_norm,basic_gl_setup,cvmat_to_glmat, draw_named_texture,create_named_texture
from glfw import *
from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D

from methods import GetAnglesPolyline,normalize
from cache_list import Cache_List

#ctypes import for atb_vars:
from ctypes import c_int,c_bool,create_string_buffer
from time import time

import logging
logger = logging.getLogger(__name__)

from reference_surface import Reference_Surface

class Offline_Reference_Surface(Reference_Surface):
    """docstring for Offline_Reference_Surface"""
    def __init__(self,g_pool,name="unnamed",saved_definition=None, gaze_positions_by_frame = None):
        super(Offline_Reference_Surface, self).__init__(name,saved_definition)
        self.g_pool = g_pool
        self.gaze_positions_by_frame = gaze_positions_by_frame
        self.cache = None
        self.gaze_on_srf = [] # points on surface for realtime feedback display

        self.heatmap_detail = .2 
        self.heatmap = None
        self.heatmap_texture = None
    #cache fn for offline marker
    def locate_from_cache(self,frame_idx):
        if self.cache == None:
            #no cache available cannot update from cache
            return False
        cache_result = self.cache[frame_idx]
        if cache_result == False:
            #cached data not avaible for this frame
            return False
        elif cache_result == None:
            #cached data shows surface not found:
            self.detected = False
            self.m_from_screen = None
            self.m_to_screen = None
            self.gaze_on_srf = []
            self.detected_markers = 0
            return True
        else:
            self.detected = True
            self.m_from_screen = cache_result['m_from_screen']
            self.m_to_screen =  cache_result['m_to_screen']       
            self.detected_markers = cache_result['detected_markers']
            self.gaze_on_srf = cache_result['gaze_on_srf']
            return True
        raise Exception("Invalid cache entry. Please report Bug.")


    def update_cache(self,marker_cache,idx=None):
        '''
        compute surface m's and gaze points from cached marker data
        entries are:
            - False: when marker cache entry was False (not yet searched)
            - None: when surface was not found
            - {'m_to_screen':,'m_from_screen':,'detected_markers':,gaze_on_srf} 
        '''

        # iterations = 0
 
        if self.cache == None:
            pass
            # self.init_cache(marker_cache)       
        elif idx != None:
            #update single data pt
            self.cache.update(idx,self.answer_caching_request(marker_cache,idx))
        else:
            # update where markercache is not False but surface cache is still false 
            # this happens when the markercache was incomplete when this fn was run before
            for i in range(len(marker_cache)):
                if self.cache[i] == False and marker_cache[i] != False:
                    self.cache.update(i,self.answer_caching_request(marker_cache,i))
                    # iterations +=1
        # return iterations



    def init_cache(self,marker_cache):
        if self.defined:
            logger.debug("Full update of surface '%s' positons cache"%self.name)
            self.cache = Cache_List([self.answer_caching_request(marker_cache,i) for i in xrange(len(marker_cache))],positive_eval_fn=lambda x:  (x!=False) and (x!=None)) 


    def answer_caching_request(self,marker_cache,frame_index):
        visible_markers = marker_cache[frame_index]
        # cache point had not been visited 
        if visible_markers == False:
            return False
        # cache point had been visited
        marker_by_id = dict( [ (m['id'],m) for m in visible_markers] )
        visible_ids = set(marker_by_id.keys())
        requested_ids = set(self.markers.keys())
        overlap = visible_ids & requested_ids
        detected_markers = len(overlap)
        if len(overlap)>=min(2,len(requested_ids)):
            yx = np.array( [marker_by_id[i]['verts_norm'] for i in overlap] )
            uv = np.array( [self.markers[i].uv_coords for i in overlap] )
            yx.shape=(-1,1,2)
            uv.shape=(-1,1,2)
            m_to_screen,mask = cv2.findHomography(uv,yx)
            m_from_screen,mask = cv2.findHomography(yx,uv)

            return {'m_to_screen':m_to_screen,
                    'm_from_screen':m_from_screen,
                    'detected_markers':len(overlap),
                    'gaze_on_srf':self.gaze_on_srf_by_frame_idx(frame_index,m_from_screen)}
        else:
            #surface not found
            return None


    def gaze_on_srf_by_frame_idx(self,frame_index,m_from_screen):
        gaze_positions = self.gaze_positions_by_frame[frame_index]
        gaze_on_src = []
        for g_p in gaze_positions:
            gaze_points = np.array([g_p['norm_gaze']]).reshape(1,1,2) 
            gaze_points_on_srf = cv2.perspectiveTransform(gaze_points , m_from_screen )
            gaze_points_on_srf.shape = (2) 
            gaze_on_src.append( {'norm_gaze_on_srf':(gaze_points_on_srf[0],gaze_points_on_srf[1]),'timestamp':g_p['timestamp'] } )
        return gaze_on_src



    #### fns to draw surface in seperate window
    def gl_display_in_window(self,world_tex_id):
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
            gluOrtho2D(0, 1, 0, 1) # gl coord convention

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            #apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            draw_named_texture(world_tex_id)

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()


            if self.heatmap_texture:
                draw_named_texture(self.heatmap_texture)

            # now lets get recent pupil positions on this surface:
            for gp in self.gaze_on_srf:
                draw_gl_points_norm([gp['norm_gaze_on_srf']],color=(0.,8.,.5,.8), size=80)
           
            glfwSwapBuffers(self._window)
            glfwMakeContextCurrent(active_window)


    def generate_heatmap(self):

        in_mark = self.g_pool.trim_marks.in_mark
        out_mark = self.g_pool.trim_marks.out_mark

        x,y = self.scale_factor
        x = max(1,int(x))
        y = max(1,int(y))

        filter_size = (int(self.heatmap_detail * x)/2)*2 +1 
        std_dev = filter_size /6.
        self.heatmap = np.ones((y,x,4),dtype=np.uint8)
        all_gaze = []
        for idx,c_e in enumerate(self.cache):
            if in_mark <= idx <= out_mark:
                if c_e:
                    for gp in c_e['gaze_on_srf']:
                        all_gaze.append(gp['norm_gaze_on_srf'])

        if not all_gaze:
            logger.warning("No gaze data on surface for heatmap found.")
            all_gaze.append((-1.,-1.))
        all_gaze = np.array(all_gaze)
        all_gaze *= self.scale_factor
        hist,xedge,yedge = np.histogram2d(all_gaze[:,0], all_gaze[:,1], 
                                            bins=[x,y], 
                                            range=[[0, self.scale_factor[0]], [0,self.scale_factor[1]]], 
                                            normed=False, 
                                            weights=None)


        hist = np.rot90(hist)

        #smoothing..
        hist = cv2.GaussianBlur(hist, (filter_size,filter_size),std_dev)
        maxval = np.amax(hist)
        if maxval:
            scale = 255./maxval
        else:
            scale = 0
        
        hist = np.uint8( hist*(scale) )

        #colormapping
        c_map = cv2.applyColorMap(hist, cv2.COLORMAP_JET)

        self.heatmap[:,:,:3] = c_map
        self.heatmap[:,:,3] = 125 
        self.heatmap_texture = create_named_texture(self.heatmap)
