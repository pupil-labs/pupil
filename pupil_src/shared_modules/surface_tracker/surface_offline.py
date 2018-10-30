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

from cache_list import Cache_List
import player_methods

from surface_tracker.surface import Surface, Surface_Location
from surface_tracker import offline_utils
from surface_tracker import background_tasks


class Surface_Offline(Surface):
    """Surface_Offline uses a cache to reuse previously computed surface locations.

    The cache is filled in the background.
    """

    def __init__(self, name="unknown", init_dict=None):
        super().__init__(name=name, init_dict=init_dict)
        self.cache_seek_idx = mp.Value("i", 0)
        self.location_cache = None
        self.location_cache_filler = None
        self.observations_frame_idxs = []
        self.on_surface_changed = None
        self.start_idx = None

    def map_section(self, section, all_world_timestamps, all_gaze_events, camera_model):
        try:
            location_cache = self.location_cache[section]
        except TypeError:
            return []

        section_gaze_on_surf = []
        for frame_idx, location in enumerate(location_cache):
            frame_idx += section.start

            if location and location.detected:

                frame_window = player_methods.enclosing_window(
                    all_world_timestamps, frame_idx
                )
                gaze_events = all_gaze_events.by_ts_window(frame_window)

                gaze_on_surf = self.map_gaze_and_fixation_events(
                    gaze_events, camera_model, trans_matrix=location.img_to_surf_trans
                )
            else:
                gaze_on_surf = []
            section_gaze_on_surf.append(gaze_on_surf)
        return section_gaze_on_surf

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

                if def_idx not in self.observations_frame_idxs:
                    markers = marker_cache[def_idx]
                    markers = {m.id: m for m in markers}
                    self._update_definition(def_idx, markers, camera_model)

                # Stop searching if we looped once through the entire recording
                if def_idx == frame_idx - 1:
                    self.build_up_status = 1.0
                    self._finalize_def()
                    break

                def_idx += 1

            if self.defined:
                # All previous detections were preliminary, devalidate them.
                self.location_cache = None
                if self.on_surface_changed is not None:
                    self.on_surface_changed(self)

        try:
            if self.location_cache_filler is not None:
                for cache_idx, location in self.location_cache_filler.fetch():
                    self.location_cache.update(cache_idx, location, force=True)

                if self.location_cache_filler.completed:
                    self.location_cache_filler = None
                    self.on_surface_changed(self)

            location = self.location_cache[frame_idx]
        except (TypeError, AttributeError):
            location = False
            self._recalculate_location_cache(frame_idx, marker_cache, camera_model)

        if location is False:
            if not marker_cache[frame_idx] is False:
                logging.debug("On demand surface cache update!")
                self.update_location_cache(frame_idx, marker_cache, camera_model)
                self.update_location(frame_idx, marker_cache, camera_model)
                return
            else:
                logging.debug("Markers not computed yet!")
                location = Surface_Location(detected=False)

        self.detected = location.detected
        self.dist_img_to_surf_trans = location.dist_img_to_surf_trans
        self.surf_to_dist_img_trans = location.surf_to_dist_img_trans
        self.img_to_surf_trans = location.img_to_surf_trans
        self.surf_to_img_trans = location.surf_to_img_trans
        self.num_detected_markers = location.num_detected_markers

    def update_location_cache(self, frame_idx, marker_cache, camera_model):
        """ Update a single entry in the location cache."""

        try:
            if not marker_cache[frame_idx]:
                location = Surface_Location(detected=False)
            else:
                markers = marker_cache[frame_idx]
                markers = {m.id: m for m in markers}
                location = Surface.locate(
                    markers,
                    camera_model,
                    self.reg_markers_undist,
                    self.reg_markers_dist,
                )
            self.location_cache.update(frame_idx, location, force=True)
        except (TypeError, AttributeError) as e:
            self._recalculate_location_cache(frame_idx, marker_cache, camera_model)

    def _recalculate_location_cache(self, frame_idx, marker_cache, camera_model):
        logging.debug("Recalculate Surface Cache!")
        if self.location_cache_filler is not None:
            self.location_cache_filler.cancel()

        # Reset cache and recalculate all entries for which previous marker detections existed.
        visited_list = [e is False for e in marker_cache]
        self.cache_seek_idx.value = frame_idx
        self.location_cache = Cache_List(
            [False] * len(marker_cache), positive_eval_fn=_cache_positive_eval_fn
        )
        self.location_cache_filler = background_tasks.background_data_processor(
            marker_cache,
            offline_utils.surface_locater_callable(
                camera_model, self.reg_markers_undist, self.reg_markers_dist
            ),
            visited_list,
            self.cache_seek_idx,
        )

    def _update_definition(self, idx, visible_markers, camera_model):
        self.observations_frame_idxs.append(idx)
        super()._update_definition(idx, visible_markers, camera_model)

    def move_corner(self, frame_idx, marker_cache, corner_idx, new_pos, camera_model):
        super().move_corner(corner_idx, new_pos, camera_model)

        # Soft reset of marker cache. This does not invoke a recalculation in the background. Full recalculation will happen once the surface corner was released.
        self.location_cache = Cache_List(
            [False] * len(marker_cache), positive_eval_fn=_cache_positive_eval_fn
        )
        self.update_location_cache(frame_idx, marker_cache, camera_model)

    def add_marker(self, marker_id, verts_px, camera_model):
        super().add_marker(marker_id, verts_px, camera_model)
        self.location_cache = None

    def pop_marker(self, id):
        super().pop_marker(id)
        self.location_cache = None

    def save_to_dict(self):
        save_dict = super().save_to_dict()
        try:
            cache_to_file = []
            for location in self.location_cache:
                if location["dist_img_to_surf_trans"] is not None:
                    location = location.copy()
                    location["dist_img_to_surf_trans"] = location[
                        "dist_img_to_surf_trans"
                    ].tolist()
                    location["surf_to_dist_img_trans"] = location[
                        "surf_to_dist_img_trans"
                    ].tolist()
                    location["img_to_surf_trans"] = location[
                        "img_to_surf_trans"
                    ].tolist()
                    location["surf_to_img_trans"] = location[
                        "surf_to_img_trans"
                    ].tolist()
                cache_to_file.append(location)
            save_dict["cache"] = cache_to_file
        except TypeError:
            save_dict["cache"] = None
        save_dict["start_idx"] = self.start_idx
        save_dict["observations_frame_idxs"] = self.observations_frame_idxs
        return save_dict

    def _load_from_dict(self, init_dict):
        super()._load_from_dict(init_dict)
        try:
            cache = init_dict["cache"]
            for location in cache:
                if location["dist_img_to_surf_trans"] is not None:
                    location["dist_img_to_surf_trans"] = np.asarray(
                        location["dist_img_to_surf_trans"]
                    )
                    location["surf_to_dist_img_trans"] = np.asarray(
                        location["surf_to_dist_img_trans"]
                    )
                    location["img_to_surf_trans"] = np.asarray(
                        location["img_to_surf_trans"]
                    )
                    location["surf_to_img_trans"] = np.asarray(
                        location["surf_to_img_trans"]
                    )
            self.location_cache = Cache_List(
                cache, positive_eval_fn=_cache_positive_eval_fn
            )
        except (KeyError, TypeError):
            self.location_cache = None

        try:
            self.observations_frame_idxs = init_dict["observations_frame_idxs"]
            self.start_idx = init_dict["start_idx"]
        except KeyError:
            # If surface was created in Capture, we just accept it as is
            self.observations_frame_idxs = []
            self.start_idx = 0
            self.build_up_status = 1.0

    def visible_count_in_section(self, section):
        """Count in how many frames the surface was visible in a section."""
        if self.location_cache is None:
            return 0
        section_cache = self.location_cache[section]
        return sum(map(bool, section_cache))


def _cache_positive_eval_fn(x):
    return (x is not False) and x.detected
