"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import itertools
import logging
import multiprocessing
import os
import platform
import time

logger = logging.getLogger(__name__)

import numpy as np
import cv2
import pyglui
import gl_utils
import pyglui.cygl.utils as pyglui_utils
import OpenGL.GL as gl

from plugin import Analysis_Plugin_Base
import file_methods

from surface_tracker.cache import Cache
from surface_tracker.surface_tracker import Surface_Tracker
from surface_tracker import offline_utils, background_tasks, Square_Marker_Detection
from surface_tracker.gui import Heatmap_Mode
from surface_tracker.surface_offline import Surface_Offline

# On macOS, "spawn" is set as default start method in main.py. This is not required
# here and we set it back to "fork" to improve performance.
if platform.system() == "Darwin":
    mp_context = multiprocessing.get_context("fork")
else:
    mp_context = multiprocessing.get_context()


class Surface_Tracker_Offline(Surface_Tracker, Analysis_Plugin_Base):
    """
    The Surface_Tracker_Offline does marker based AOI tracking in a recording. All
    marker and surface detections are calculated in the background and cached to reduce
    computation.
    """

    order = 0.2
    TIMELINE_LINE_HEIGHT = 16

    def __init__(self, g_pool, marker_min_perimeter=60, inverted_markers=False):
        super().__init__(g_pool, marker_min_perimeter, inverted_markers)

        self.MARKER_CACHE_VERSION = 3
        # Also add very small detected markers to cache and filter cache afterwards
        self.CACHE_MIN_MARKER_PERIMETER = 20
        self.cache_seek_idx = mp_context.Value("i", 0)
        self.marker_cache = None
        self.marker_cache_unfiltered = None
        self.cache_filler = None
        self._init_marker_cache()
        self.last_cache_update_ts = time.time()
        self.CACHE_UPDATE_INTERVAL_SEC = 5

        self.gaze_on_surf_buffer = None
        self.gaze_on_surf_buffer_filler = None

        self._heatmap_update_requests = set()
        self.export_proxies = set()

    @property
    def Surface_Class(self):
        return Surface_Offline

    @property
    def _save_dir(self):
        return self.g_pool.rec_dir

    @property
    def has_freeze_feature(self):
        return False

    @property
    def ui_info_text(self):
        return (
            "The offline surface tracker will look for markers in the entire "
            "video. By default it uses surfaces defined in capture. You can "
            "change and add more surfaces here. \n \n Press the export button or "
            "type 'e' to start the export."
        )

    @property
    def supported_heatmap_modes(self):
        return [Heatmap_Mode.WITHIN_SURFACE, Heatmap_Mode.ACROSS_SURFACES]

    def _init_marker_cache(self):
        previous_cache = file_methods.Persistent_Dict(
            os.path.join(self.g_pool.rec_dir, "square_marker_cache")
        )
        version = previous_cache.get("version", 0)
        cache = previous_cache.get("marker_cache_unfiltered", None)

        if cache is None:
            self._recalculate_marker_cache()
        elif version != self.MARKER_CACHE_VERSION:
            logger.debug("Marker cache version missmatch. Rebuilding marker cache.")
            self.inverted_markers = previous_cache.get("inverted_markers", False)
            self._recalculate_marker_cache()
        else:
            marker_cache_unfiltered = []
            for markers in cache:
                # Loaded markers are either False, [] or a list of dictionaries. We
                # need to convert the dictionaries into Square_Marker_Detection objects.
                if markers:

                    markers = [
                        Square_Marker_Detection(*args) if args else None
                        for args in markers
                    ]
                marker_cache_unfiltered.append(markers)

            self._recalculate_marker_cache(previous_state=marker_cache_unfiltered)
            self.inverted_markers = previous_cache.get("inverted_markers", False)
            logger.debug("Restored previous marker cache.")

    def _recalculate_marker_cache(self, previous_state=None):
        if previous_state is None:
            previous_state = [None for _ in self.g_pool.timestamps]

            # If we had a previous_state argument, surface objects had just been
            # initialized with their previous state, which we do not want to overwrite.
            # Therefore resetting the marker cache is only done when no previous_state
            # is defined.
            for surface in self.surfaces:
                surface.location_cache = None

        self.marker_cache_unfiltered = Cache(previous_state)
        self.marker_cache = self._filter_marker_cache(self.marker_cache_unfiltered)

        self.cache_filler = background_tasks.background_video_processor(
            self.g_pool.capture.source_path,
            offline_utils.marker_detection_callable(
                self.CACHE_MIN_MARKER_PERIMETER, self.inverted_markers
            ),
            list(self.marker_cache),
            self.cache_seek_idx,
            mp_context,
        )

    def _filter_marker_cache(self, cache_to_filter):
        marker_cache = []
        for markers in cache_to_filter:
            if markers:
                markers = self._filter_markers(markers)
            marker_cache.append(markers)
        return Cache(marker_cache)

    def init_ui(self):
        super().init_ui()

        self.glfont = pyglui.pyfontstash.fontstash.Context()
        self.glfont.add_font("opensans", pyglui.ui.get_opensans_font_path())
        self.glfont.set_color_float((1.0, 1.0, 1.0, 0.8))
        self.glfont.set_align_string(v_align="right", h_align="top")

        self.timeline = pyglui.ui.Timeline(
            "Surface Tracker",
            self._gl_display_cache_bars,
            self._draw_labels,
            self.TIMELINE_LINE_HEIGHT * (len(self.surfaces) + 1),
        )
        self.g_pool.user_timelines.append(self.timeline)
        self.timeline.content_height = (
            len(self.surfaces) + 1
        ) * self.TIMELINE_LINE_HEIGHT

    def recent_events(self, events):
        super().recent_events(events)
        self._fetch_data_from_bg_fillers()

    def _fetch_data_from_bg_fillers(self):
        if self.gaze_on_surf_buffer_filler is not None:
            for gaze in self.gaze_on_surf_buffer_filler.fetch():
                self.gaze_on_surf_buffer.append(gaze)

            if self.gaze_on_surf_buffer_filler.completed:
                self.gaze_on_surf_buffer_filler = None
                self._update_surface_heatmaps()
                self.gaze_on_surf_buffer = None

        for proxy in list(self.export_proxies):
            for _ in proxy.fetch():
                pass

            if proxy.completed:
                self.export_proxies.remove(proxy)

    def _update_markers(self, frame):
        self._update_marker_and_surface_caches()

        self.markers = self.marker_cache[frame.index]
        self.markers_unfiltered = self.marker_cache_unfiltered[frame.index]
        if self.markers is None:
            # Move seek index to current frame because caches do not contain data for it
            self.markers = []
            self.markers_unfiltered = []
            self.cache_seek_idx.value = frame.index

    def _update_marker_and_surface_caches(self):
        if self.cache_filler is None:
            return

        for frame_index, markers in self.cache_filler.fetch():
            if frame_index is None:
                continue
            markers = self._remove_duplicate_markers(markers)
            self.marker_cache_unfiltered.update(frame_index, markers)
            markers_filtered = self._filter_markers(markers)
            self.marker_cache.update(frame_index, markers_filtered)

            for surface in self.surfaces:
                surface.update_location_cache(
                    frame_index, self.marker_cache, self.camera_model
                )

        if self.cache_filler.completed:
            self.cache_filler = None
            for surface in self.surfaces:
                self._heatmap_update_requests.add(surface)
            self._fill_gaze_on_surf_buffer()
            self._save_marker_cache()
            self.save_surface_definitions_to_file()

        now = time.time()
        if now - self.last_cache_update_ts > self.CACHE_UPDATE_INTERVAL_SEC:
            self._save_marker_cache()
            self.last_cache_update_ts = now

    def _update_surface_locations(self, frame_index):
        for surface in self.surfaces:
            surface.update_location(frame_index, self.marker_cache, self.camera_model)

    def _update_surface_corners(self):
        for surface, corner_idx in self._edit_surf_verts:
            if surface.detected:
                surface.move_corner(
                    self.current_frame.index,
                    self.marker_cache,
                    corner_idx,
                    self._last_mouse_pos.copy(),
                    self.camera_model,
                )

    def _update_surface_heatmaps(self):
        self._compute_across_surfaces_heatmap()

        for surface in self._heatmap_update_requests:
            surf_idx = self.surfaces.index(surface)
            gaze_on_surf = self.gaze_on_surf_buffer[surf_idx]
            gaze_on_surf = list(itertools.chain.from_iterable(gaze_on_surf))
            surface.update_heatmap(gaze_on_surf)

        self._heatmap_update_requests.clear()

    def _compute_across_surfaces_heatmap(self):
        gaze_counts_per_surf = []
        for gaze in self.gaze_on_surf_buffer:
            gaze = list(itertools.chain.from_iterable(gaze))
            gaze = [g for g in gaze if g["on_surf"]]
            gaze_counts_per_surf.append(len(gaze))

        if gaze_counts_per_surf:
            max_count = max(gaze_counts_per_surf)
            results = np.array(gaze_counts_per_surf, dtype=np.float32)
            if max_count > 0:
                results *= 255.0 / max_count
            results = np.uint8(results)
            results_color_maps = cv2.applyColorMap(results, cv2.COLORMAP_JET)

            for surface, color_map in zip(self.surfaces, results_color_maps):
                heatmap = np.ones((1, 1, 4), dtype=np.uint8) * 125
                heatmap[:, :, :3] = color_map
                surface.across_surface_heatmap = heatmap
        else:
            for surface in self.surfaces:
                surface.across_surface_heatmap = surface.get_uniform_heatmap()

    def _fill_gaze_on_surf_buffer(self):
        in_mark = self.g_pool.seek_control.trim_left
        out_mark = self.g_pool.seek_control.trim_right
        section = slice(in_mark, out_mark)

        all_world_timestamps = self.g_pool.timestamps
        all_gaze_events = self.g_pool.gaze_positions

        self._start_gaze_buffer_filler(all_gaze_events, all_world_timestamps, section)

    def _start_gaze_buffer_filler(self, all_gaze_events, all_world_timestamps, section):
        if self.gaze_on_surf_buffer_filler is not None:
            self.gaze_on_surf_buffer_filler.cancel()
        self.gaze_on_surf_buffer = []
        self.gaze_on_surf_buffer_filler = background_tasks.background_gaze_on_surface(
            self.surfaces,
            section,
            all_world_timestamps,
            all_gaze_events,
            self.camera_model,
            mp_context,
        )

    def _start_fixation_buffer_filler(
        self, all_fixation_events, all_world_timestamps, section
    ):
        if self.fixations_on_surf_buffer_filler is not None:
            self.fixations_on_surf_buffer_filler.cancel()
        self.fixations_on_surf_buffer = []
        self.fixations_on_surf_buffer_filler = background_tasks.background_gaze_on_surface(
            self.surfaces,
            section,
            all_world_timestamps,
            all_fixation_events,
            self.camera_model,
            mp_context,
        )

    def gl_display(self):
        if self.timeline:
            self.timeline.refresh()
        super().gl_display()

    def _gl_display_cache_bars(self, width, height, scale):
        ts = self.g_pool.timestamps
        with gl_utils.Coord_System(ts[0], ts[-1], height, 0):
            # Lines for areas that have been cached
            cached_ranges = []
            for r in self.marker_cache.visited_ranges:
                cached_ranges += ((ts[r[0]], 0), (ts[r[1]], 0))

            gl.glTranslatef(0, scale * self.TIMELINE_LINE_HEIGHT / 2, 0)
            color = pyglui_utils.RGBA(0.8, 0.2, 0.2, 0.8)
            pyglui_utils.draw_polyline(
                cached_ranges, color=color, line_type=gl.GL_LINES, thickness=scale * 4
            )
            cached_ranges = []
            for r in self.marker_cache.positive_ranges:
                cached_ranges += ((ts[r[0]], 0), (ts[r[1]], 0))

            color = pyglui_utils.RGBA(0, 0.7, 0.3, 0.8)
            pyglui_utils.draw_polyline(
                cached_ranges, color=color, line_type=gl.GL_LINES, thickness=scale * 4
            )

            # Lines where surfaces have been found in video
            cached_surfaces = []
            for surface in self.surfaces:
                found_at = []
                if surface.location_cache is not None:
                    for r in surface.location_cache.positive_ranges:  # [[0,1],[3,4]]
                        found_at += ((ts[r[0]], 0), (ts[r[1]], 0))
                cached_surfaces.append(found_at)

            color = pyglui_utils.RGBA(0, 0.7, 0.3, 0.8)

            for surface in cached_surfaces:
                gl.glTranslatef(0, scale * self.TIMELINE_LINE_HEIGHT, 0)
                pyglui_utils.draw_polyline(
                    surface, color=color, line_type=gl.GL_LINES, thickness=scale * 2
                )

    def _draw_labels(self, width, height, scale):
        self.glfont.set_size(self.TIMELINE_LINE_HEIGHT * 0.8 * scale)
        self.glfont.draw_text(width, 0, "Marker Cache")
        for surface in self.surfaces:
            gl.glTranslatef(0, self.TIMELINE_LINE_HEIGHT * scale, 0)
            self.glfont.draw_text(width, 0, surface.name)

    def add_surface(self, init_dict=None):
        super().add_surface(init_dict)

        try:
            self.timeline.content_height += self.TIMELINE_LINE_HEIGHT
            self._fill_gaze_on_surf_buffer()
        except AttributeError:
            pass
        self.surfaces[-1].on_surface_change = self.on_surface_change

    def remove_surface(self, surface):
        super().remove_surface(surface)
        try:
            self._heatmap_update_requests.remove(surface)
        except KeyError:
            pass
        self.timeline.content_height -= self.TIMELINE_LINE_HEIGHT

    def on_notify(self, notification):
        super().on_notify(notification)

        if notification["subject"] == "surface_tracker.marker_detection_params_changed":
            self._recalculate_marker_cache()

        elif notification["subject"] == "surface_tracker.marker_min_perimeter_changed":
            self.marker_cache = self._filter_marker_cache(self.marker_cache_unfiltered)
            for surface in self.surfaces:
                surface.location_cache = None

        elif notification["subject"] == "surface_tracker.heatmap_params_changed":
            for surface in self.surfaces:
                if surface.name == notification["name"]:
                    self._heatmap_update_requests.add(surface)
                    surface.within_surface_heatmap = surface.get_placeholder_heatmap()
                    break
            self._fill_gaze_on_surf_buffer()

        elif notification["subject"].startswith("seek_control.trim_indices_changed"):
            for surface in self.surfaces:
                surface.within_surface_heatmap = surface.get_placeholder_heatmap()
                self._heatmap_update_requests.add(surface)
            self._fill_gaze_on_surf_buffer()

        elif notification["subject"] == "surface_tracker.surfaces_changed":
            for surface in self.surfaces:
                if surface.name == notification["name"]:
                    surface.location_cache = None
                    surface.within_surface_heatmap = surface.get_placeholder_heatmap()
                    self._heatmap_update_requests.add(surface)
                    break

        elif notification["subject"] == "should_export":
            proxy = background_tasks.get_export_proxy(
                notification["export_dir"],
                notification["range"],
                self.surfaces,
                self.g_pool.timestamps,
                self.g_pool.gaze_positions,
                self.g_pool.fixations,
                self.camera_model,
                mp_context,
            )
            self.export_proxies.add(proxy)

        elif notification["subject"] == "gaze_positions_changed":
            for surface in self.surfaces:
                self._heatmap_update_requests.add(surface)
                surface.within_surface_heatmap = surface.get_placeholder_heatmap()
            self._fill_gaze_on_surf_buffer()

    def on_surface_change(self, surface):
        self.save_surface_definitions_to_file()
        self._heatmap_update_requests.add(surface)
        self._fill_gaze_on_surf_buffer()

    def deinit_ui(self):
        super().deinit_ui()
        self.g_pool.user_timelines.remove(self.timeline)
        self.timeline = None
        self.glfont = None

    def cleanup(self):
        super().cleanup()
        self._save_marker_cache()

        for proxy in self.export_proxies:
            proxy.cancel()
            self.export_proxies.remove(proxy)

    def _save_marker_cache(self):
        marker_cache_file = file_methods.Persistent_Dict(
            os.path.join(self.g_pool.rec_dir, "square_marker_cache")
        )
        marker_cache_file["marker_cache_unfiltered"] = list(
            self.marker_cache_unfiltered
        )
        marker_cache_file["version"] = self.MARKER_CACHE_VERSION
        marker_cache_file["inverted_markers"] = self.inverted_markers
        marker_cache_file.save()
