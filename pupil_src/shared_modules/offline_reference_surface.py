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
from gl_utils import cvmat_to_glmat
from glfw import *
from OpenGL.GL import *
from pyglui.cygl.utils import Named_Texture
from cache_list import Cache_List
from reference_surface import Reference_Surface

import logging
logger = logging.getLogger(__name__)


class Offline_Reference_Surface(Reference_Surface):
    """docstring for Offline_Reference_Surface"""
    def __init__(self, g_pool, name="unnamed",saved_definition=None):
        super().__init__(g_pool, name, saved_definition)
        self.g_pool = g_pool
        self.cache = None

        self.metrics_gazecount = None
        self.metrics_texture = None

    # cache fn for offline marker
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
            self.gaze_on_srf = self.gaze_on_srf_by_frame_idx(frame_idx,self.m_from_screen)
            return True
        raise Exception("Invalid cache entry. Please report Bug.")

    def update_cache(self, marker_cache, min_marker_perimeter, min_id_confidence, idx=None):
        '''
        compute surface m's and gaze points from cached marker data
        entries are:
            - False: when marker cache entry was False (not yet searched)
            - None: when surface was not found
            - {'m_to_screen':,'m_from_screen':,'detected_markers':,gaze_on_srf}
        '''

        # iterations = 0
        if self.cache is None:
            pass
            # self.init_cache(marker_cache)
        elif idx is not None:
            # update single data pt
            self.cache.update(idx, self.answer_caching_request(marker_cache, idx, min_marker_perimeter, min_id_confidence))
        else:
            # update where marker cache is not False but surface cache is still false
            # this happens when the markercache was incomplete when this fn was run before
            for i in range(len(marker_cache)):
                if self.cache[i] is False and marker_cache[i] is not False:
                    self.cache.update(i, self.answer_caching_request(marker_cache, i, min_marker_perimeter, min_id_confidence))
                    # iterations +=1
        # return iterations

    def init_cache(self, marker_cache, min_marker_perimeter, min_id_confidence):
        if self.defined:
            logger.debug("Full update of surface '{}' positons cache".format(self.name))
            self.cache = Cache_List([self.answer_caching_request(marker_cache, i, min_marker_perimeter, min_id_confidence) for i in range(len(marker_cache))], positive_eval_fn=lambda x:  (x is not False) and (x is not None))

    def answer_caching_request(self, marker_cache, frame_index, min_marker_perimeter, min_id_confidence):
        visible_markers = marker_cache[frame_index]
        # cache point had not been visited
        if visible_markers is False:
            return False
        res = self._get_location(visible_markers, min_marker_perimeter, min_id_confidence, locate_3d=False)
        if res['detected']:
            return res
        else:
            # surface not found
            return None

    def move_vertex(self, vert_idx, new_pos):
        super().move_vertex(vert_idx,new_pos)
        self.cache = None

    def add_marker(self, marker, visible_markers, min_marker_perimeter, min_id_confidence):
        super().add_marker(marker, visible_markers, min_marker_perimeter, min_id_confidence)
        self.cache = None

    def remove_marker(self, marker):
        super().remove_marker(marker)
        self.cache = None

    def gaze_on_srf_by_frame_idx(self, frame_index, m_from_screen):
        return self.map_data_to_surface(self.g_pool.gaze_positions_by_frame[frame_index], m_from_screen)

    def fixations_on_srf_by_frame_idx(self, frame_index, m_from_screen):
        return self.map_data_to_surface(self.g_pool.fixations_by_frame[frame_index], m_from_screen)

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

    def generate_heatmap(self, section):
        if self.cache is None:
            logger.warning('Surface cache is not build yet.')
            return

        all_gaze = []
        for frame_idx, c_e in enumerate(self.cache[section]):
            if c_e:
                frame_idx += section.start
                for gp in self.gaze_on_srf_by_frame_idx(frame_idx, c_e['m_from_screen']):
                    if gp['confidence'] >= self.g_pool.min_data_confidence:
                        all_gaze.append(gp['norm_pos'])

        self._generate_heatmap(all_gaze)

    def visible_count_in_section(self,section):
        #section is a slice
        #return number of frames where surface is visible.
        #If cache is not available on frames it is reported as not visible
        if self.cache is None:
            return 0
        section_cache = self.cache[section]
        return sum(map(bool,section_cache))

    def gaze_on_srf_in_section(self,section=slice(0,None)):
        #section is a slice
        #return number of gazepoints that are on surface in section
        #If cache is not available on frames it is reported as not visible
        if self.cache is None:
            return []
        gaze_on_srf = []
        for frame_idx,c_e in enumerate(self.cache[section]):
            frame_idx+=section.start
            if c_e:
                gaze_on_srf += [gp for gp in self.gaze_on_srf_by_frame_idx(frame_idx,c_e['m_from_screen']) if gp['on_srf']]
        return gaze_on_srf

