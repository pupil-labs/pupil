"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import multiprocessing as mp
from .surface import Surface
from . import offline_utils

import numpy as np
import cv2
from gl_utils import cvmat_to_glmat
from glfw import *
from OpenGL.GL import *
from pyglui.cygl.utils import Named_Texture
from cache_list import Cache_List
from reference_surface import Reference_Surface
import player_methods as pm

import logging

logger = logging.getLogger(__name__)

# TODO clean up imports
from .background_tasks import background_data_processor


class Offline_Surface(Surface):
    def __init__(self, init_dict=None):
        super().__init__(init_dict=init_dict)
        self.cache_seek_idx = mp.Value("i", 0)
        self.cache = None
        self.cache_filler = None
        self.observations_frame_idxs = []

    def recalculate_cache(self, marker_cache, camera_model):
        if self.cache_filler is not None:
            self.cache_filler.cancel()

        # Reset cache and recalculate all entriesfor which marker detections exist.
        visited_list = [not e for e in marker_cache]
        self.cache = Cache_List([False] * len(marker_cache))
        self.cache_filler = background_data_processor(
            marker_cache,
            offline_utils.surface_locater_callable(camera_model, self.reg_markers_undist, self.reg_markers_dist),
            visited_list,
            self.cache_seek_idx,
        )

    def move_corner(self, corner_idx, pos, camera_model):
        super().move_corner(corner_idx, pos, camera_model)
        self.cache = None

    def add_marker(self, id, verts, camera_model):
        super().add_marker(id, verts, camera_model)
        self.cache = None

    def pop_marker(self, id):
        super().pop_marker(id)
        self.cache = None

    def update_location(self, idx, marker_cache, camera_model):

        if not self.defined:
            try:
                frame_idx = self.start_idx
            except AttributeError:
                frame_idx = self.start_idx = idx

            while not self.defined:
                # End of the video, start from the beginning!
                if frame_idx == len(marker_cache) and idx > 0:
                    frame_idx = 0

                if marker_cache[frame_idx] is False:
                    break

                if not frame_idx in self.observations_frame_idxs:
                    markers = marker_cache[frame_idx]
                    markers = {m.id: m for m in markers}
                    self.update_def(frame_idx, markers, camera_model)

                # Stop searching if we looped once through the entire recording
                if frame_idx == idx - 1:
                    self.build_up_status = 1.0
                    self._finalize_def()
                    break

                frame_idx += 1

            if self.defined:
                # All previous detections were preliminary, devalidate them.
                self.cache = None

        try:
            if self.cache_filler is not None:
                for idx, location in self.cache_filler.fetch():
                    self.cache.update(idx, location)
            location = self.cache[idx]
        except (TypeError, AttributeError):
            location = False
            self.cache_seek_idx.value = idx
            self.recalculate_cache(marker_cache, camera_model)

        if not location:
            location = {}
            location["detected"] = False
            location["dist_img_to_surf_trans"] = None
            location["surf_to_dist_img_trans"] = None
            location["img_to_surf_trans"] = None
            location["surf_to_img_trans"] = None
            location["num_detected_markers"] = 0

        self.__dict__.update(location)

    def update_cache(self, idx, marker_cache, camera_model):
        """
        Use cached marker data to update the surface cache.
        The surface cache contains the following values:
            - False: if the corresponding marker cache entry is False (not yet
            searched).
            - None: if the surface was not found.
            - Dict containing all image-surface transformations and gaze on surface
            values: otherwise
        """

        try:
            if self.cache[idx] is None:
                raise Exception("Should never be reached!")  # TODO delete this case
                return
            elif not marker_cache[idx]:
                location = {}
                location["detected"] = False
                location["dist_img_to_surf_trans"] = None
                location["surf_to_dist_img_trans"] = None
                location["img_to_surf_trans"] = None
                location["surf_to_img_trans"] = None
                location["num_detected_markers"] = 0
            else:
                markers = marker_cache[idx]
                markers = {m.id: m for m in markers}
                location = Surface.locate(
                    markers,
                    camera_model,
                    self.reg_markers_undist,
                    self.reg_markers_dist,
                )
            self.cache.update(idx, location)
        except TypeError:
            self.recalculate_cache(marker_cache, camera_model)

    def update_def(self, idx, vis_markers, camera_model):
        self.observations_frame_idxs.append(idx)
        super().update_def(idx, vis_markers, camera_model)

    # TODO Why?
    def gaze_on_srf_by_frame_idx(self, frame_index, m_from_screen):
        frame_window = pm.enclosing_window(self.g_pool.timestamps, frame_index)
        return self.map_data_to_surface(
            self.g_pool.gaze_positions.by_ts_window(frame_window), m_from_screen
        )

    # TODO Why?
    def fixations_on_srf_by_frame_idx(self, frame_index, m_from_screen):
        frame_window = pm.enclosing_window(self.g_pool.timestamps, frame_index)
        return self.map_data_to_surface(
            self.g_pool.fixations.by_ts_window(frame_window), m_from_screen
        )

    # TODO check
    def gl_display_metrics(self):
        if self.metrics_texture and self.detected:
            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_to_screen)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, 1, 0, 1, -1, 1)  # gl coord convention

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            # apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            self.metrics_texture.draw()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
