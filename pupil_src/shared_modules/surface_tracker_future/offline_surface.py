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
import logging
logger = logging.getLogger(__name__)

import numpy as np
from gl_utils import cvmat_to_glmat
from OpenGL.GL import *

from cache_list import Cache_List
import player_methods as pm

from .surface import Surface
from . import offline_utils
from . import background_tasks


class Offline_Surface(Surface):
    def __init__(self, on_surface_changed=None, init_dict=None):
        self.cache_seek_idx = mp.Value("i", 0)
        self.cache = None
        self.cache_filler = None
        self.observations_frame_idxs = []
        self.current_frame_idx = None
        super().__init__(on_surface_changed=on_surface_changed, init_dict=init_dict)

    def recalculate_cache(self, frame_idx, marker_cache, camera_model):
        logging.debug("Recaclulate Surface Cache!")
        if self.cache_filler is not None:
            self.cache_filler.cancel()

        # Reset cache and recalculate all entries for which previous marker detections existed.
        visited_list = [e is False for e in marker_cache]
        self.cache_seek_idx.value = frame_idx
        self.cache = Cache_List([False] * len(marker_cache), positive_eval_fn=lambda x: (x is not False) and x['detected'])
        self.cache_filler = background_tasks.background_data_processor(
            marker_cache,
            offline_utils.surface_locater_callable(camera_model, self.reg_markers_undist, self.reg_markers_dist),
            visited_list,
            self.cache_seek_idx,
        )

    def move_corner(self, frame_idx, marker_cache, corner_idx, pos, camera_model):
        super().move_corner(corner_idx, pos, camera_model)

        # Soft reset of marker cache. This does not invoke a recalculation in the background. Full recalculation will happen once the surface corner was released.
        self.cache = Cache_List([False] * len(marker_cache), positive_eval_fn=lambda x: (x is not False) and x['detected'])
        self.update_cache(frame_idx, marker_cache, camera_model)

    def add_marker(self, id, verts, camera_model):
        super().add_marker(id, verts, camera_model)
        self.cache = None

    def pop_marker(self, id):
        super().pop_marker(id)
        self.cache = None

    def update_location(self, frame_idx, marker_cache, camera_model):
        if not self.defined:
            try:
                def_idx = self.start_idx
            except AttributeError:
                def_idx = self.start_idx = frame_idx

            while not self.defined:
                # End of the video, start from the beginning!
                if def_idx == len(marker_cache) and frame_idx > 0:
                    def_idx = 0

                if marker_cache[def_idx] is False:
                    break

                if not def_idx in self.observations_frame_idxs:
                    markers = marker_cache[def_idx]
                    markers = {m.id: m for m in markers}
                    self.update_def(def_idx, markers, camera_model)

                # Stop searching if we looped once through the entire recording
                if def_idx == frame_idx - 1:
                    self.build_up_status = 1.0
                    self._finalize_def()
                    break

                def_idx += 1

            if self.defined:
                # All previous detections were preliminary, devalidate them.
                self.cache = None
                if self.on_surface_change is not None:
                    self.on_surface_change()

        try:
            if self.cache_filler is not None:
                for cache_idx, location in self.cache_filler.fetch():
                    self.cache.update(cache_idx, location, force=True)

                if self.cache_filler.completed:
                    self.cache_filler = None
                    self.on_surface_change()

            location = self.cache[frame_idx]
        except (TypeError, AttributeError):
            location = False
            self.recalculate_cache(frame_idx, marker_cache, camera_model)

        if location is False:
            if not marker_cache[frame_idx] is False:
                logging.debug("On demand surface cache update!")
                self.update_cache(frame_idx, marker_cache, camera_model)
                self.update_location(frame_idx, marker_cache, camera_model)
                return
            else:
                logging.debug("Markers not computed yet!")
                location = {}
                location["detected"] = False
                location["dist_img_to_surf_trans"] = None
                location["surf_to_dist_img_trans"] = None
                location["img_to_surf_trans"] = None
                location["surf_to_img_trans"] = None
                location["num_detected_markers"] = 0

        self.__dict__.update(location)

    def update_cache(self, frame_idx, marker_cache, camera_model):
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
            if not marker_cache[frame_idx]:
                location = {}
                location["detected"] = False
                location["dist_img_to_surf_trans"] = None
                location["surf_to_dist_img_trans"] = None
                location["img_to_surf_trans"] = None
                location["surf_to_img_trans"] = None
                location["num_detected_markers"] = 0
            else:
                markers = marker_cache[frame_idx]
                markers = {m.id: m for m in markers}
                location = Surface.locate(
                    markers,
                    camera_model,
                    self.reg_markers_undist,
                    self.reg_markers_dist,
                )
            self.cache.update(frame_idx, location, force=True)
        except (TypeError, AttributeError):
            self.recalculate_cache(frame_idx, marker_cache, camera_model)

    def update_def(self, idx, vis_markers, camera_model):
        self.observations_frame_idxs.append(idx)
        super().update_def(idx, vis_markers, camera_model)

    def on_change(self):
        # TODO is it nicer to call recalculate_cache directly?
        self.cache = None

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

    def save_to_dict(self):
        save_dict = super().save_to_dict()
        try:
            cache_to_file = []
            for location in self.cache:
                if location["dist_img_to_surf_trans"] is not None:
                    location = location.copy()
                    location["dist_img_to_surf_trans"] = location["dist_img_to_surf_trans"].tolist()
                    location["surf_to_dist_img_trans"] = location["surf_to_dist_img_trans"].tolist()
                    location["img_to_surf_trans"] = location["img_to_surf_trans"].tolist()
                    location["surf_to_img_trans"] = location["surf_to_img_trans"].tolist()
                cache_to_file.append(location)
            save_dict['cache'] = cache_to_file
        except TypeError:
            save_dict['cache'] = None
        save_dict['start_idx'] = self.start_idx
        save_dict['observations_frame_idxs'] = self.observations_frame_idxs
        return save_dict

    def load_from_dict(self, init_dict):
        super().load_from_dict(init_dict)
        try:
            cache = init_dict['cache']
            for location in cache:
                if location["dist_img_to_surf_trans"] is not None:
                    location["dist_img_to_surf_trans"] = np.asarray(location["dist_img_to_surf_trans"])
                    location["surf_to_dist_img_trans"] = np.asarray(location["surf_to_dist_img_trans"])
                    location["img_to_surf_trans"] = np.asarray(location["img_to_surf_trans"])
                    location["surf_to_img_trans"] = np.asarray(location["surf_to_img_trans"])
            self.cache = Cache_List(cache, positive_eval_fn=lambda x: (x is not False) and x['detected'])
        except (TypeError, AttributeError):
            self.cache = None

        try:
            self.observations_frame_idxs= init_dict['observations_frame_idxs']
            self.start_idx = init_dict['start_idx']
        except AttributeError:
            self.observations_frame_idxs = []
