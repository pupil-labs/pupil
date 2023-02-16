"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import multiprocessing
import platform

import player_methods

from . import background_tasks, offline_utils
from .cache import Cache
from .surface import Surface, Surface_Location

logger = logging.getLogger(__name__)


# On macOS, "spawn" is set as default start method in main.py. This is not required
# here and we set it back to "fork" to improve performance.
if platform.system() == "Darwin":
    mp_context = multiprocessing.get_context("fork")
else:
    mp_context = multiprocessing.get_context()


class Surface_Offline(Surface):
    """Surface_Offline uses a cache to reuse previously computed surface locations.

    The cache is filled in the background.
    """

    def __init__(self, *args, **kwargs):
        self.location_cache = None
        super().__init__(*args, **kwargs)
        self.cache_seek_idx = mp_context.Value("i", 0)
        self.location_cache_filler = None
        self.observations_frame_idxs = []
        self.on_surface_change = None
        self.start_idx = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["on_surface_change"]
        del state["location_cache_filler"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

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
            self._build_definition_from_cache(camera_model, frame_idx, marker_cache)

        self._fetch_from_location_cache_filler()
        try:
            location = self.location_cache[frame_idx]
        except (TypeError, AttributeError):
            # If any event invalidates the location_cache, it will be set to None.
            location = None
            self._recalculate_location_cache(frame_idx, marker_cache, camera_model)

        # If location is None the cache was not filled at the current position yet.
        if location is None:
            if not marker_cache[frame_idx] is None:
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

    def _build_definition_from_cache(self, camera_model, frame_idx, marker_cache):
        def_idx = self.start_idx
        while not self.defined:
            # End of the video, start from the beginning!
            if def_idx == len(marker_cache) and frame_idx > 0:
                def_idx = 0

            try:
                if marker_cache[def_idx] is False:
                    break
            except TypeError:
                # start_idx was not yet defined! Current frame will become first
                # frame to define this surface.
                def_idx = self.start_idx = frame_idx

            if def_idx not in self.observations_frame_idxs:
                markers = marker_cache[def_idx]
                markers = {m.uid: m for m in markers}
                self._update_definition(def_idx, markers, camera_model)

            # Stop searching if we looped once through the entire recording
            if def_idx == frame_idx - 1:
                self.build_up_status = 1.0
                self.prune_markers()
                break

            def_idx += 1

        else:
            # All previous detections were preliminary, devalidate them.
            self.location_cache = None
            if self.on_surface_change is not None:
                self.on_surface_change(self)

    def _fetch_from_location_cache_filler(self):
        if self.location_cache_filler is not None:
            for cache_idx, location in self.location_cache_filler.fetch():
                try:
                    self.location_cache.update(cache_idx, location, force=True)
                except AttributeError:
                    self.location_cache_filler.cancel()
                    self.location_cache_filler = None
                    return

            if self.location_cache_filler.completed:
                self.location_cache_filler = None
                self.on_surface_change(self)

    def update_location_cache(
        self, frame_idx, marker_cache, camera_model, context=None
    ):
        """Update a single entry in the location cache."""

        try:
            if not marker_cache[frame_idx]:
                location = Surface_Location(detected=False)
            else:
                markers = marker_cache[frame_idx]
                markers = {m.uid: m for m in markers}
                location = Surface.locate(
                    markers,
                    camera_model,
                    self.registered_markers_undist,
                    self.registered_markers_dist,
                    context=context,
                )
            self.location_cache.update(frame_idx, location, force=True)
        except (TypeError, AttributeError):
            self._recalculate_location_cache(frame_idx, marker_cache, camera_model)

    def _recalculate_location_cache(self, frame_idx, marker_cache, camera_model):
        logging.debug("Recalculate Surface Cache!")
        if self.location_cache_filler is not None:
            self.location_cache_filler.cancel()

        # Reset cache and recalculate.
        self.cache_seek_idx.value = frame_idx
        self.location_cache = Cache([None for _ in marker_cache])
        self.location_cache_filler = background_tasks.background_data_processor(
            marker_cache,
            offline_utils.surface_locater_callable(
                camera_model,
                self.registered_markers_undist,
                self.registered_markers_dist,
            ),
            self.cache_seek_idx,
            mp_context,
        )

    def _update_definition(self, idx, visible_markers, camera_model):
        self.observations_frame_idxs.append(idx)
        super()._update_definition(idx, visible_markers, camera_model)

    def move_corner(self, frame_idx, marker_cache, corner_idx, new_pos, camera_model):
        super().move_corner(corner_idx, new_pos, camera_model)

        # Reset of marker cache. This does not invoke a recalculation in the background.
        # Full recalculation will happen once the surface corner was released.
        self.location_cache = Cache([None for _ in marker_cache])
        self.update_location_cache(frame_idx, marker_cache, camera_model)

    def add_marker(self, marker_id, verts_px, camera_model):
        super().add_marker(marker_id, verts_px, camera_model)
        self.location_cache = None

    def pop_marker(self, id):
        super().pop_marker(id)
        self.location_cache = None

    def save_to_dict(self):
        save_dict = super().save_to_dict()
        if self.location_cache is None:
            cache_to_file = None
        else:
            cache_to_file = []
            for location in self.location_cache:
                if location is None:
                    # We do not save partial marker caches
                    cache_to_file = None
                    break
                else:
                    location_searializable = location.get_serializable_copy()
                cache_to_file.append(location_searializable)
        save_dict["cache"] = cache_to_file

        save_dict["added_in_player"] = {
            "start_idx": self.start_idx,
            "observations_frame_idxs": self.observations_frame_idxs,
        }
        return save_dict

    def _load_from_dict(self, init_dict):
        super()._load_from_dict(init_dict)
        try:
            cache = init_dict["cache"]
            for cache_idx in range(len(cache)):
                location = cache[cache_idx]
                cache[cache_idx] = Surface_Location.load_from_serializable_copy(
                    location
                )

            self.location_cache = Cache(cache)
        except (KeyError, TypeError):
            self.location_cache = None

        try:
            added_in_player = init_dict["added_in_player"]
        except KeyError:
            # If surface was created in Capture, we just accept it as is
            self.observations_frame_idxs = []
            self.start_idx = 0
            self.build_up_status = 1.0
        else:
            self.observations_frame_idxs = added_in_player["observations_frame_idxs"]
            self.start_idx = added_in_player["start_idx"]

    def visible_count_in_section(self, section):
        """Count in how many frames the surface was visible in a section."""
        if self.location_cache is None:
            return 0
        section_cache = self.location_cache[section]
        return sum(map(bool, section_cache))
