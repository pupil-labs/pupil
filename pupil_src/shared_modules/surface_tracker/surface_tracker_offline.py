"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

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
import tempfile
import time
import typing as T

import cv2
import data_changed
import file_methods
import gl_utils
import numpy as np
import OpenGL.GL as gl
import pyglui
import pyglui.cygl.utils as pyglui_utils
from observable import Observable
from plugin import Plugin

from . import background_tasks, offline_utils
from .cache import Cache
from .gui import Heatmap_Mode
from .surface_marker import Surface_Marker
from .surface_marker_detector import MarkerDetectorMode, MarkerType
from .surface_offline import Surface_Offline
from .surface_tracker import (
    APRILTAG_HIGH_RES_ON,
    APRILTAG_SHARPENING_ON,
    Surface_Tracker,
)

logger = logging.getLogger(__name__)


# On macOS, "spawn" is set as default start method in main.py. This is not required
# here and we set it back to "fork" to improve performance.
if platform.system() == "Darwin":
    mp_context = multiprocessing.get_context("fork")
else:
    mp_context = multiprocessing.get_context()


class _CacheRelevantDetectorParams(T.NamedTuple):
    mode: MarkerDetectorMode
    inverted_markers: bool
    quad_decimate: float
    sharpening: float


class Surface_Tracker_Offline(Observable, Surface_Tracker, Plugin):
    """
    The Surface_Tracker_Offline does marker based AOI tracking in a recording. All
    marker and surface detections are calculated in the background and cached to reduce
    computation.
    """

    order = 0.2
    TIMELINE_LINE_HEIGHT = 16

    def __init__(self, g_pool, *args, **kwargs):
        super().__init__(g_pool, *args, use_online_detection=False, **kwargs)

        self.MARKER_CACHE_VERSION = 3
        # Also add very small detected markers to cache and filter cache afterwards
        self.CACHE_MIN_MARKER_PERIMETER = 20
        self.cache_seek_idx = mp_context.Value("i", 0)
        self.marker_cache = None
        self.marker_cache_unfiltered = None
        self.cache_filler = None
        self._init_marker_cache()
        self.last_cache_update_ts = time.perf_counter()
        self.CACHE_UPDATE_INTERVAL_SEC = 5

        self.gaze_on_surf_buffer = None
        self.gaze_on_surf_buffer_filler = None

        self._heatmap_update_requests = set()
        self.export_proxies = set()

        self._gaze_changed_listener = data_changed.Listener(
            "gaze_positions", g_pool.rec_dir, plugin=self
        )
        self._gaze_changed_listener.add_observer(
            "on_data_changed", self._on_gaze_positions_changed
        )

        self.__surface_location_context = {}

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

    @staticmethod
    def _marker_detector_mode_from_cache(
        marker_cache,
    ) -> T.Optional[MarkerDetectorMode]:
        assert marker_cache is not None

        # Filter out non-filled frames where the cache entry is None.
        # Chain the remaining entries (which are lists) to get a flat sequence.
        filled_out_marker_cache = filter(lambda x: x is not None, marker_cache)
        cached_surface_marker_sequence = itertools.chain.from_iterable(
            filled_out_marker_cache
        )

        # Get the first surface marker from the sequence, and set the detection mode
        # according to it.
        first_cached_surface_marker_args = next(cached_surface_marker_sequence, None)
        if first_cached_surface_marker_args is not None:
            marker = Surface_Marker.deserialize(first_cached_surface_marker_args)
            marker_detector_mode = MarkerDetectorMode.from_marker(marker)
            return marker_detector_mode

    def _init_marker_cache(self):
        previous_cache_config = file_methods.Persistent_Dict(
            os.path.join(self.g_pool.rec_dir, "square_marker_cache")
        )
        version = previous_cache_config.get("version", 0)

        previous_params = self._cache_relevant_params_from_cache(previous_cache_config)
        current_params = self._cache_relevant_params_from_controller()

        if previous_params is None:
            self._recalculate_marker_cache(parameters=current_params)
        elif version != self.MARKER_CACHE_VERSION:
            logger.debug("Marker cache version missmatch. Rebuilding marker cache.")
            self._recalculate_marker_cache(parameters=current_params)
        else:
            marker_cache_unfiltered = []
            for markers in previous_cache_config["marker_cache_unfiltered"]:
                # Loaded markers are either False, [] or a list of dictionaries. We
                # need to convert the dictionaries into Surface_Marker objects.
                if markers:

                    markers = [
                        Surface_Marker.deserialize(args) if args else None
                        for args in markers
                    ]
                marker_cache_unfiltered.append(markers)

            self._recalculate_marker_cache(
                parameters=previous_params, previous_state=marker_cache_unfiltered
            )
            logger.debug("Restored previous marker cache.")

    def _set_detector_params(self, params: _CacheRelevantDetectorParams):
        self.marker_detector._marker_detector_mode = params.mode
        self.marker_detector._square_marker_inverted_markers = params.inverted_markers
        self.marker_detector._apriltag_quad_decimate = params.quad_decimate
        self.marker_detector._apriltag_decode_sharpening = params.sharpening
        self.marker_detector.init_detector()

    @staticmethod
    def _cache_relevant_params_from_cache(
        previous_cache,
    ) -> T.Optional[_CacheRelevantDetectorParams]:
        marker_cache_unfiltered = previous_cache.get("marker_cache_unfiltered", None)
        if marker_cache_unfiltered is None:
            return

        mode = Surface_Tracker_Offline._marker_detector_mode_from_cache(
            marker_cache_unfiltered
        )
        if mode is None:
            return

        return _CacheRelevantDetectorParams(
            mode=mode,
            inverted_markers=previous_cache.get("inverted_markers", False),
            quad_decimate=previous_cache.get("quad_decimate", APRILTAG_HIGH_RES_ON),
            sharpening=previous_cache.get("sharpening", APRILTAG_SHARPENING_ON),
        )

    def _cache_relevant_params_from_controller(self) -> _CacheRelevantDetectorParams:
        return _CacheRelevantDetectorParams(
            mode=self.marker_detector.marker_detector_mode,
            inverted_markers=self.marker_detector.inverted_markers,
            quad_decimate=self.marker_detector.apriltag_quad_decimate,
            sharpening=self.marker_detector.apriltag_decode_sharpening,
        )

    def _recalculate_marker_cache(
        self, parameters: _CacheRelevantDetectorParams, previous_state=None
    ):
        # Ensures consistency across foreground and background detectors
        self._set_detector_params(parameters)
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

        if self.cache_filler is not None:
            self.cache_filler.cancel()

        self.__surface_location_context = {}
        self.cache_filler = background_tasks.background_video_processor(
            self.g_pool.capture.source_path,
            offline_utils.marker_detection_callable.from_detector(
                self.marker_detector, self.CACHE_MIN_MARKER_PERIMETER
            ),
            list(self.marker_cache),
            self.cache_seek_idx,
            mp_context,
        )

    def _filter_marker_cache(self, cache_to_filter):
        marker_type = self.marker_detector.marker_detector_mode.marker_type
        if marker_type != MarkerType.SQUARE_MARKER:
            # We only need to filter SQUARE_MARKERs
            return cache_to_filter

        marker_cache = []
        for markers in cache_to_filter:
            if markers:
                markers = self._filter_markers(markers)
            marker_cache.append(markers)
        return Cache(marker_cache)

    def _filter_markers(self, markers):
        return [
            m
            for m in markers
            if m.perimeter >= self.marker_detector.marker_min_perimeter
        ]

    def init_ui(self):
        super().init_ui()

        self.glfont = pyglui.pyfontstash.fontstash.Context()
        self.glfont.add_font("opensans", pyglui.ui.get_opensans_font_path())
        self.glfont.set_color_float((1.0, 1.0, 1.0, 0.8))
        self.glfont.set_align_string(v_align="right", h_align="top")

        self.timeline = pyglui.ui.Timeline(
            "Surface Tracker",
            self._timeline_draw_data_cb,
            self._timeline_draw_label_cb,
            self.TIMELINE_LINE_HEIGHT * (len(self.surfaces) + 1),
        )
        self.g_pool.user_timelines.append(self.timeline)
        self.timeline.content_height = (
            len(self.surfaces) + 1
        ) * self.TIMELINE_LINE_HEIGHT
        self._set_timeline_refresh_needed()

    def recent_events(self, events):
        super().recent_events(events)
        self._fetch_data_from_bg_fillers()

    def _fetch_data_from_bg_fillers(self):
        if self.gaze_on_surf_buffer_filler is not None:
            start_time = time.perf_counter()
            did_timeout = False

            for gaze in self.gaze_on_surf_buffer_filler.fetch():
                self.gaze_on_surf_buffer.append(gaze)
                if time.perf_counter() - start_time > 1 / 50:
                    did_timeout = True
                    break

            if self.gaze_on_surf_buffer_filler.completed and not did_timeout:
                self.gaze_on_surf_buffer_filler = None
                self._update_surface_heatmaps()
                self.gaze_on_surf_buffer = None

            self._set_timeline_refresh_needed()

        for proxy in list(self.export_proxies):
            for _ in proxy.fetch():
                pass

            if proxy.completed:
                self.export_proxies.remove(proxy)

    def _update_markers(self, frame):
        self._update_marker_and_surface_caches()

        self.markers = self.marker_cache[frame.index]
        if self.markers is None:
            # Move seek index to current frame because caches do not contain data for it
            self.markers = []
            self.cache_seek_idx.value = frame.index

    def _update_marker_and_surface_caches(self):
        if self.cache_filler is None:
            return

        start_time = time.perf_counter()
        did_timeout = False

        for frame_index, markers in self.cache_filler.fetch():
            if frame_index is not None:
                markers = self._remove_duplicate_markers(markers)
                self.marker_cache_unfiltered.update(frame_index, markers)
                marker_type = self.marker_detector.marker_detector_mode.marker_type
                if marker_type == MarkerType.SQUARE_MARKER:
                    markers_filtered = self._filter_markers(markers)
                    self.marker_cache.update(frame_index, markers_filtered)
                # In all other cases (see _filter_marker_cache()):
                # `self.marker_cache is self.marker_cache_unfiltered == True`

                for surface in self.surfaces:
                    surface.update_location_cache(
                        frame_index,
                        self.marker_cache,
                        self.camera_model,
                        context=self.__surface_location_context,
                    )
            if time.perf_counter() - start_time > 1 / 50:
                did_timeout = True
                break

        if self.cache_filler.completed and not did_timeout:
            self.cache_filler = None
            for surface in self.surfaces:
                self._heatmap_update_requests.add(surface)
            self._fill_gaze_on_surf_buffer()
            self._save_marker_cache()
            self.save_surface_definitions_to_file()

        now = time.perf_counter()
        if now - self.last_cache_update_ts > self.CACHE_UPDATE_INTERVAL_SEC:
            self._save_marker_cache()
            self.last_cache_update_ts = now

        self._set_timeline_refresh_needed()

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
            gaze_on_surf = itertools.chain.from_iterable(gaze_on_surf)
            gaze_on_surf = (
                g
                for g in gaze_on_surf
                if g["confidence"] >= self.g_pool.min_data_confidence
            )
            gaze_on_surf = list(gaze_on_surf)
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
                surface.across_surface_heatmap = surface.get_uniform_heatmap((1, 1))

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

    def gl_display(self):
        self._timeline_refresh_if_needed()
        super().gl_display()

    def _set_timeline_refresh_needed(self):
        self.__is_timeline_refresh_needed = True

    def _timeline_refresh_if_needed(self):
        try:
            is_timeline_refresh_needed = self.__is_timeline_refresh_needed
        except AttributeError:
            return
        if is_timeline_refresh_needed and self.timeline:
            self.__is_timeline_refresh_needed = False
            self.timeline.refresh()

    def _timeline_draw_data_cb(self, width, height, scale):
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

    def _timeline_draw_label_cb(self, width, height, scale):
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
        self._set_timeline_refresh_needed()

    def remove_surface(self, surface):
        super().remove_surface(surface)
        try:
            self._heatmap_update_requests.remove(surface)
        except KeyError:
            pass
        self.timeline.content_height -= self.TIMELINE_LINE_HEIGHT
        self._set_timeline_refresh_needed()

    def on_notify(self, notification):
        super().on_notify(notification)

        if notification["subject"] == "surface_tracker.marker_detection_params_changed":
            current_params = self._cache_relevant_params_from_controller()
            self._recalculate_marker_cache(parameters=current_params)

        elif notification["subject"] == "surface_tracker.marker_min_perimeter_changed":
            marker_type = self.marker_detector.marker_detector_mode.marker_type
            if marker_type == MarkerType.SQUARE_MARKER:
                self.marker_cache = self._filter_marker_cache(
                    self.marker_cache_unfiltered
                )
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

            if self.cache_filler is not None:
                logger.error("Marker detection not finished. No data will be exported.")
                return

            if self.gaze_on_surf_buffer_filler is not None:
                logger.error(
                    "Surface gaze mapping not finished. No data will be exported."
                )
                return

            # Create new marker cache temporary file
            # Backgroud exporter is responsible of removing the temporary file when finished
            file_handle, marker_cache_path = tempfile.mkstemp()
            os.close(file_handle)  # https://bugs.python.org/issue42830

            # Save marker cache into the new temporary file
            temp_marker_cache = file_methods.Persistent_Dict(marker_cache_path)
            temp_marker_cache["marker_cache"] = self.marker_cache
            temp_marker_cache.save()

            proxy = background_tasks.get_export_proxy(
                notification["export_dir"],
                notification["range"],
                self.surfaces,
                self.g_pool.timestamps,
                self.g_pool.gaze_positions,
                self.g_pool.fixations,
                self.camera_model,
                marker_cache_path,
                mp_context,
            )
            self.export_proxies.add(proxy)

        elif (
            notification["subject"]
            == "surface_tracker_offline._should_fill_gaze_on_surf_buffer"
        ):
            self._fill_gaze_on_surf_buffer()

    def _on_gaze_positions_changed(self):
        for surface in self.surfaces:
            self._heatmap_update_requests.add(surface)
            surface.within_surface_heatmap = surface.get_placeholder_heatmap()
        self._fill_gaze_on_surf_buffer()

    def on_surface_change(self, surface):
        self.save_surface_definitions_to_file()
        self._heatmap_update_requests.add(surface)
        self._debounced_fill_gaze_on_surf_buffer()

    def _debounced_fill_gaze_on_surf_buffer(self):
        self.notify_all(
            {
                "subject": "surface_tracker_offline._should_fill_gaze_on_surf_buffer",
                "delay": 1.0,
            }
        )

    def deinit_ui(self):
        super().deinit_ui()
        self.g_pool.user_timelines.remove(self.timeline)
        self.timeline = None
        self.glfont = None

    def cleanup(self):
        super().cleanup()
        self._save_marker_cache()

        for proxy in self.export_proxies.copy():
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

        current_config = self._cache_relevant_params_from_controller()
        marker_cache_file["inverted_markers"] = current_config.inverted_markers
        marker_cache_file["quad_decimate"] = current_config.quad_decimate
        marker_cache_file["sharpening"] = current_config.sharpening
        marker_cache_file.save()
