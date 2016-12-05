'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import numpy as np
import cv2
from gl_utils import cvmat_to_glmat,clear_gl_screen
from glfw import *
from OpenGL.GL import *
from pyglui.cygl.utils import Named_Texture, draw_points_norm, RGBA
from methods import GetAnglesPolyline,normalize
from cache_list import Cache_List

#ctypes import for atb_vars:
from ctypes import c_int,c_bool,create_string_buffer
from time import time

import logging
logger = logging.getLogger(__name__)

from reference_surface import Reference_Surface, m_verts_to_screen, m_verts_from_screen

class Offline_Reference_Surface(Reference_Surface):
    """docstring for Offline_Reference_Surface"""
    def __init__(self,g_pool,name="unnamed",saved_definition=None):
        super(Offline_Reference_Surface, self).__init__(name,saved_definition)
        self.g_pool = g_pool
        self.raw_cache = None
        self.smoothed_cache = None
        self.should_smooth = saved_definition.get('should_smooth',True)
        self.gaze_on_srf = [] # points on surface for realtime feedback display

        self.heatmap_detail = .2
        self.heatmap = None
        self.heatmap_texture = None
        self.metrics_gazecount = None
        self.metrics_texture = None

        self.cov_threshold = 4.
        self.cov_distribution = None

    def setup_kalman_filter(self, n_dim_obs):
        f = super(Offline_Reference_Surface, self).setup_kalman_filter(n_dim_obs)
        Q = 0.5 * np.eye(self.kfilter_state_dim)
        f.transition_covariance = Q
        return f

    def clear_smoothing(self):
        self.smoothed_cache = None

    def refresh_smoothing(self):
        if not self.raw_cache or not self.should_smooth:
            return # editing surface, smooth later, or should not smooth at all
        self.kfilter = self.setup_kalman_filter(n_dim_obs=8)
        raw_data = np.ma.masked_all((self.raw_cache.length, self.kfilter_obs_dim))
        for i,entry in enumerate(self.raw_cache):
            if entry:
                raw_data[i,:] = self.create_kalman_oberservation(entry['raw_location'])
        # self.kfilter.em(raw_data,n_iter=5)
        smoothed_mean, smoothed_cov = self.kfilter.smooth(raw_data)
        self.cov_distribution = smoothed_cov.sum(axis=(1,2))
        projected = np.dot(smoothed_mean, self.kfilter.observation_matrices.T)
        projected = projected.reshape(-1,4,2).astype(np.float32)
        smoothed = [None] * self.cov_distribution.shape[0]
        for idx in xrange(self.cov_distribution.shape[0]):
            if self.cov_distribution[idx] < self.cov_threshold:

                detected_markers = 0
                if self.raw_cache[idx]:
                    detected_markers = self.raw_cache[idx]['detected_markers']
                smoothed[idx] = {
                    'detected'        :True,
                    'detected_markers':detected_markers,
                    'm_from_screen'   :m_verts_from_screen(projected[idx,:,:]),
                    'm_to_screen'     :m_verts_to_screen(projected[idx,:,:]),
                    'summed_kfilter_cov':self.cov_distribution[idx]
                }
            else:
                smoothed[idx] = {
                    'detected'        :False,
                    'detected_markers':0,
                    'm_from_screen'   :None,
                    'm_to_screen'     :None,
                    'summed_kfilter_cov':None
                }
        eval_fn = lambda entry: entry['detected']
        self.smoothed_cache = Cache_List(smoothed, positive_eval_fn=eval_fn)

    #cache fn for offline marker
    def locate_from_cache(self,frame_idx):
        if self.smoothed_cache:
            cache_result = self.smoothed_cache[frame_idx]
            if cache_result == False:
                pass
            else:
                self.detected = cache_result['detected']
                self.m_from_screen = cache_result['m_from_screen']
                self.m_to_screen =  cache_result['m_to_screen']
                self.detected_markers = cache_result['detected_markers']
                self.current_cov =  cache_result['summed_kfilter_cov']
                self.gaze_on_srf = self.gaze_on_srf_by_frame_idx(frame_idx,self.m_from_screen) if self.m_from_screen is not None else None
                return True

        self.current_cov = None
        if self.raw_cache == None:
            #no cache available cannot update from cache
            return False
        cache_result = self.raw_cache[frame_idx]
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
            self.gaze_on_srf = self.gaze_on_srf_by_frame_idx(frame_idx,self.m_from_screen)
            return True
        raise Exception("Invalid cache entry. Please report Bug.")


    def update_cache(self,marker_cache,camera_calibration,min_marker_perimeter,min_id_confidence,idx=None):
        '''
        compute surface m's and gaze points from cached marker data
        entries are:
            - False: when marker cache entry was False (not yet searched)
            - None: when surface was not found
            - {'m_to_screen':,'m_from_screen':,'detected_markers':,gaze_on_srf}
        '''

        # iterations = 0
        if self.raw_cache == None:
            pass
            # self.init_cache(marker_cache)
        elif idx != None:
            #update single data pt
            self.raw_cache.update(idx,self.answer_caching_request(marker_cache,idx,camera_calibration,min_marker_perimeter,min_id_confidence))
        else:
            # update where marker cache is not False but surface cache is still false
            # this happens when the markercache was incomplete when this fn was run before
            for i in range(len(marker_cache)):
                if self.raw_cache[i] == False and marker_cache[i] != False:
                    self.raw_cache.update(i,self.answer_caching_request(marker_cache,i,camera_calibration,min_marker_perimeter,min_id_confidence))
                    # iterations +=1
        # return iterations



    def init_cache(self,marker_cache,camera_calibration,min_marker_perimeter,min_id_confidence):
        if self.defined:
            logger.debug("Full update of surface '%s' positons cache"%self.name)
            self.raw_cache = Cache_List([self.answer_caching_request(marker_cache,i,camera_calibration,min_marker_perimeter,min_id_confidence) for i in xrange(len(marker_cache))],positive_eval_fn=lambda x:  (x!=False) and (x!=None))


    def answer_caching_request(self,marker_cache,frame_index,camera_calibration,min_marker_perimeter,min_id_confidence):
        visible_markers = marker_cache[frame_index]
        # cache point had not been visited
        if visible_markers == False:
            return False
        res = self._get_location(visible_markers,camera_calibration,min_marker_perimeter,min_id_confidence,locate_3d=False,smooth=False)
        if res['detected']:
            return res
        else:
            #surface not found
            return None

    def move_vertex(self,vert_idx,new_pos):
        super(Offline_Reference_Surface, self).move_vertex(vert_idx,new_pos)
        self.raw_cache = None
        self.smoothed_cache = None
        self.heatmap = None

    def add_marker(self,marker,visible_markers,camera_calibration,min_marker_perimeter,min_id_confidence):
        super(Offline_Reference_Surface, self).add_marker(marker,visible_markers,camera_calibration,min_marker_perimeter,min_id_confidence)
        self.raw_cache = None
        self.smoothed_cache = None
        self.heatmap = None

    def remove_marker(self,marker):
        super(Offline_Reference_Surface, self).remove_marker(marker)
        self.raw_cache = None
        self.smoothed_cache = None
        self.heatmap = None

    def gaze_on_srf_by_frame_idx(self,frame_index,m_from_screen):
        return self.map_data_to_surface(self.g_pool.gaze_positions_by_frame[frame_index],m_from_screen)

    def fixations_on_srf_by_frame_idx(self,frame_index,m_from_screen):
        return self.map_data_to_surface(self.g_pool.fixations_by_frame[frame_index],m_from_screen)

    def gl_display_heatmap(self):
        if self.heatmap_texture and self.detected:

            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_to_screen)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, 1, 0, 1,-1,1) # gl coord convention

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            #apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            self.heatmap_texture.draw()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()


    def gl_display_metrics(self):
        if self.metrics_texture and self.detected:


            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_to_screen)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, 1, 0, 1,-1,1) # gl coord convention

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            #apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            self.metrics_texture.draw()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()



    def generate_heatmap(self,section):

        if self.raw_cache is None:
            logger.warning('Surface cache is not build yet.')
            return


        x,y = self.real_world_size['x'],self.real_world_size['y']
        x = max(1,int(x))
        y = max(1,int(y))

        filter_size = (int(self.heatmap_detail * x)/2)*2 +1
        std_dev = filter_size /6.
        self.heatmap = np.ones((y,x,4),dtype=np.uint8)
        all_gaze = []

        for frame_idx,c_e in enumerate(self.raw_cache[section]):
            if c_e:
                frame_idx+=section.start
                for gp in self.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen']):
                    if gp['confidence']>=self.g_pool.min_data_confidence:
                        all_gaze.append(gp['norm_pos'])

        if not all_gaze:
            logger.warning("No gaze data on surface for heatmap found.")
            all_gaze.append((-1.,-1.))
        all_gaze = np.array(all_gaze)
        all_gaze *= [self.real_world_size['x'],self.real_world_size['y']]
        hist,xedge,yedge = np.histogram2d(all_gaze[:,0], all_gaze[:,1],
                                            bins=[x,y],
                                            range=[[0, self.real_world_size['x']], [0,self.real_world_size['y']]],
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
        self.heatmap_texture = Named_Texture()
        self.heatmap_texture.update_from_ndarray(self.heatmap)


    def visible_count_in_section(self,section):
        #section is a slice
        #return number of frames where surface is visible.
        #If cache is not available on frames it is reported as not visible
        if self.raw_cache is None:
            return 0
        section_cache = self.raw_cache[section]
        return sum(map(bool,section_cache))

    def gaze_on_srf_in_section(self,section=slice(0,None)):
        #section is a slice
        #return number of gazepoints that are on surface in section
        #If cache is not available on frames it is reported as not visible
        if self.raw_cache is None:
            return []
        gaze_on_srf = []
        for frame_idx,c_e in enumerate(self.raw_cache[section]):
            frame_idx+=section.start
            if c_e:
                gaze_on_srf += [gp for gp in self.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen']) if gp['on_srf']]
        return gaze_on_srf

    def save_to_dict(self):
        d = super(Offline_Reference_Surface,self).save_to_dict()
        d['should_smooth'] = self.should_smooth
        return d

