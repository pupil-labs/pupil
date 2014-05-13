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
from gl_utils import draw_gl_polyline,adjust_gl_view,draw_gl_polyline_norm,clear_gl_screen,draw_gl_point,draw_gl_points,draw_gl_point_norm,draw_gl_points_norm,basic_gl_setup,cvmat_to_glmat, draw_named_texture
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
    def __init__(self,name="unnamed",saved_definition=None, gaze_positions_by_frame = None):
        super(Offline_Reference_Surface, self).__init__(name,saved_definition)
        self.gaze_positions_by_frame = gaze_positions_by_frame
        self.cache = None


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
            return True
        else:
            self.detected = True
            self.m_from_screen = cache_result['m_from_screen']
            self.m_to_screen =  cache_result['m_to_screen']       
            self.detected_markers = cache_result['detected_markers']
            self.gaze_on_srf = [gp['norm_gaze_on_srf'] for gp in cache_result['gaze_on_srf'] ]
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
            self.init_cache(marker_cache)       
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


    # def map_gaze_by_frame_onto_srf(self,gaze_by_frame):
    #     '''
    #     this fn is not cached and can be slow, dont call every frame...
    #     '''

    #     gaze_on_srf_by_frame = [[] for i in gaze_by_frame]

    #     if self.cache:
    #         for surface_data,gaze_points,result in zip(self.cache,gaze_by_frame,gaze_on_srf_by_frame):
    #             if surface_data and gaze_points:
    #                 gaze_points = np.array([d['norm_gaze'] for d in gaze_points if d is not None])
    #                 gaze_points.shape = (-1,1,2) 
    #                 gaze_points_on_srf = cv2.perspectiveTransform(gaze_points , surface_data['m_from_screen'] )
    #                 gaze_points_on_srf.shape = (-1,2) 
    #                 result.extend(gaze_points_on_srf.tolist())
    #     return gaze_on_srf_by_frame


    def save_surface_positions_to_file(self,s):
        if s.cache == None:
            logger.warning("The surface is not cached. Please wait for the cacher to collect data.")
            return

        srf_dir = os.path.join(self.g_pool.rec_dir,'surface_data',s.name.replace('/',''),s.uid)
        logger.info("exporting surface gaze data to %s"%srf_dir)
        if os.path.isdir(srf_dir):
            logger.info("Will overwrite previous export for this referece surface")
        else:
            try:
                os.mkdir(srf_dir)
            except:
                logger.warning("Could name make export dir %s"%srf_dir)
                return

        # logger.info("Saving surface positon data and gaze on surface data for '%s' with uid:'%'"%(s.name,s.uid))
        # #save surface_positions as pickle file
        # save_object(s.cache,os.path.join(srf_dir,'srf_positons_by_frame'))
        # #save surface_positions as csv
        # with open(os.path.join(srf_dir,'srf_positons_by_frame.csv','wb')) as csvfile:
        #     csw_writer =csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     csw_writer.writerow(('frame_idx','timestamp','m_to_screen','m_from_screen','detected_markers'))
        #     for idx,ts,ref_srf_data in zip(range(len(self.g_pool.timestamps)),self.g_pool.timestamps,s.cache)
        #         if ref_srf_data is not None:
        #             csw_writer.writerow( (idx,ts,ref_srf_data['m_to_screen'],ref_srf_data['m_from_screen'],ref_srf_data['detected_markers']) )
        # #save gaze on srf as csv and pickle file
        # gaze_on_ref
        # with open(os.path.join(srf_dir,'gaze_positions_on_surface','wb')) as csvfile:
        #     csw_writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     csw_writer.writerow(('world_frame_idx','world_timestamp','eye_timestamp','x_norm','y_norm','x_scaled','y_scaled'))
        #     for idx,ts,ref_srf_data in zip(range(len(self.g_pool.timestamps)),self.g_pool.timestamps,s.cache)
        #         if ref_srf_data is not None:
        #             csw_writer.writerow((idx,ts,ref_srf_data['m_to_screen'],ref_srf_data['m_from_screen'],ref_srf_data['detected_markers'])
        # #save gaze on srf as csv and pickle file