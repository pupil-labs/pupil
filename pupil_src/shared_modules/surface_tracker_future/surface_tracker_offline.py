"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import os
import logging
import time
import platform
import multiprocessing

logger = logging.getLogger(__name__)
if platform.system() == "Darwin":
    mp = multiprocessing.get_context("fork")
else:
    mp = multiprocessing.get_context()

import numpy as np
import cv2
import pyglui
import gl_utils
import pyglui.cygl.utils as pyglui_utils
import OpenGL.GL as gl

from plugin import Analysis_Plugin_Base
import file_methods
from cache_list import Cache_List

from surface_tracker_future.surface_tracker import Surface_Tracker_Future
from surface_tracker_future import offline_utils, background_tasks, Marker, Heatmap_Mode
from surface_tracker_future.surface_offline import Surface_Offline


# TODO Improve all docstrings, make methods privat appropriately
# Two colors in timeline to indicate detected markrs vs frames without markers


class Surface_Tracker_Offline_Future(Surface_Tracker_Future, Analysis_Plugin_Base):
    """
    - Mostly extends the Surface Tracker with a cache
    Special version of surface tracker for use with videofile source.
    It uses a seperate process to search all frames in the world video file for markers.
     - self.cache is a list containing marker positions for each frame.
     - self.surfaces[i].cache is a list containing surface positions for each frame
    Both caches are build up over time. The marker cache is also session persistent.
    See marker_tracker.py for more info on this marker tracker.
    """

    # TODO add surfaces export

    def __init__(self, g_pool, marker_min_perimeter=60, inverted_markers=False):
        self.timeline_line_height = 16
        self.Surface_Class = Surface_Offline
        super().__init__(g_pool, marker_min_perimeter, inverted_markers)
        self.ui_info_text = "The offline surface tracker will look for markers in the entire video. By default it uses surfaces defined in capture. You can change and add more surfaces here. \n \n Press the export button or type 'e' to start the export."
        self.supported_heatmap_modes = [
            Heatmap_Mode.WITHIN_SURFACE,
            Heatmap_Mode.ACROSS_SURFACES,
        ]

        self.order = .2

        # Caches
        self.marker_cache_version = 3
        self.cache_min_marker_perimeter = 20  # find even super small markers. The
        # surface locater will filter using min_marker_perimeter
        self.cache_seek_idx = mp.Value("i", 0)
        self._init_marker_cache()
        self.last_cache_update_ts = time.time()
        self.cache_update_interval = 5
        # self.recalculate() # What does this do?

    @property
    def save_dir(self):
        return self.g_pool.rec_dir

    def _init_marker_cache(self):
        previous_cache = file_methods.Persistent_Dict(
            os.path.join(self.g_pool.rec_dir, "square_marker_cache")
        )
        version = previous_cache.get("version", 0)
        cache = previous_cache.get("marker_cache_unfiltered", None)

        if cache is None:
            self.recalculate_marker_cache()
        elif version != self.marker_cache_version:
            logger.debug("Marker cache version missmatch. Rebuilding marker cache.")
            self.inverted_markers = previous_cache.get("inverted_markers", False)
            self.recalculate_marker_cache()
        else:
            self.marker_cache_unfiltered = []
            for markers in cache:
                if markers:
                    markers = [Marker(*args) if args else None for args in markers]
                    self.marker_cache_unfiltered.append(markers)
                else:
                    self.marker_cache_unfiltered.append(markers)
            marker_cache_unfiltered = Cache_List(self.marker_cache_unfiltered)
            self.recalculate_marker_cache(previous_state=marker_cache_unfiltered)
            self.inverted_markers = previous_cache.get("inverted_markers", False)
            logger.debug("Restored previous marker cache.")

    def recalculate_marker_cache(self, previous_state=None):
        if previous_state is None:
            previous_state = [False for _ in self.g_pool.timestamps]

            # If we had a previous_state argument, surface objects had just been
            # initialized with their previous state, which we do not want to overwrite.
            # Therefore resetting the marker cache is only done when no previous_state
            # is defined.
            for surface in self.surfaces:
                surface.cache = None

        self.marker_cache_unfiltered = Cache_List(previous_state)
        self._update_filtered_markers()

        self.cache_filler = background_tasks.background_video_processor(
            self.g_pool.capture.source_path,
            offline_utils.marker_detection_callable(
                self.cache_min_marker_perimeter, self.inverted_markers
            ),
            list(self.marker_cache),
            self.cache_seek_idx,
        )

    def _update_filtered_markers(self):
        marker_cache = []
        for markers in self.marker_cache_unfiltered:
            if markers:
                markers = [
                    m
                    for m in markers
                    if m.perimeter >= self.marker_min_perimeter
                    and m.id_confidence >= self.marker_min_confidence
                ]
            marker_cache.append(markers)
        self.marker_cache = Cache_List(marker_cache)

    def init_ui(self):
        super().init_ui()

        self.glfont = pyglui.pyfontstash.fontstash.Context()
        self.glfont.add_font("opensans", pyglui.ui.get_opensans_font_path())
        self.glfont.set_color_float((1., 1., 1., .8))
        self.glfont.set_align_string(v_align="right", h_align="top")

        self.timeline = pyglui.ui.Timeline(
            "Surface Tracker",
            self.gl_display_cache_bars,
            self.draw_labels,
            self.timeline_line_height * (len(self.surfaces) + 1),
        )
        self.g_pool.user_timelines.append(self.timeline)
        self.timeline.content_height = (
            len(self.surfaces) + 1
        ) * self.timeline_line_height

    def per_surface_ui(self, surface):
        def set_x(val):
            surface.real_world_size["x"] = val
            self.notify_all(
                {
                    "subject": "surface_tracker.heatmap_params_changed.{}".format(
                        surface.name
                    ),
                    "uid": surface.uid,
                }
            )

        def set_y(val):
            surface.real_world_size["y"] = val
            self.notify_all(
                {
                    "subject": "surface_tracker.heatmap_params_changed.{}".format(
                        surface.name
                    ),
                    "uid": surface.uid,
                }
            )

        idx = self.surfaces.index(surface)
        s_menu = pyglui.ui.Growing_Menu("Surface {}".format(idx))
        s_menu.collapsed = True
        s_menu.append(pyglui.ui.Text_Input("name", surface))
        s_menu.append(
            pyglui.ui.Text_Input(
                "x", surface.real_world_size, label="X size", setter=set_x
            )
        )
        s_menu.append(
            pyglui.ui.Text_Input(
                "y", surface.real_world_size, label="Y size", setter=set_y
            )
        )
        s_menu.append(
            pyglui.ui.Button(
                "Open Debug Window", self.gui.surface_windows[surface].open_close_window
            )
        )
        remove_surf = lambda: self.remove_surface(idx)
        s_menu.append(pyglui.ui.Button("remove", remove_surf))
        self.menu.append(s_menu)

    def _compute_across_surfaces_heatmap(
        self, section, all_gaze_timestamps, all_gaze_events
    ):
        results = []
        for s in self.surfaces:
            gaze_on_surf = s.map_section(
                section, all_gaze_timestamps, all_gaze_events, self.camera_model
            )
            results.append(len(gaze_on_surf))
            # self.metrics_gazecount = len(gaze_on_surf)

        if results == []:
            logger.warning("No surfaces defined.")
            return

        max_res = max(results)
        results = np.array(results, dtype=np.float32)
        if max_res > 0:
            results *= 255. / max_res
        results = np.uint8(results)
        results_c_maps = cv2.applyColorMap(results, cv2.COLORMAP_JET)

        for s, c_map in zip(self.surfaces, results_c_maps):
            heatmap = np.ones((1, 1, 4), dtype=np.uint8) * 125
            heatmap[:, :, :3] = c_map
            s.across_surface_heatmap = heatmap

    def update_markers(self, frame):
        if not self.cache_filler is None:
            self._update_marker_and_surface_caches()
        # Move seek index to current frame if caches do not contain data for it
        self.markers = self.marker_cache[frame.index]
        if self.markers is False:
            self.markers = []
            self.cache_seek_idx.value = frame.index

    def _update_marker_and_surface_caches(self):
        for idx, markers in self.cache_filler.fetch():
            self.marker_cache_unfiltered.update(idx, markers)
            markers_filtered = self._filter_markers(markers)
            self.marker_cache.update(idx, markers_filtered)

            for surf in self.surfaces:
                surf.update_cache(idx, self.marker_cache, self.camera_model)

        if self.cache_filler.completed:
            self.cache_filler = None
            self._save_marker_cache()
            self.save_surface_definitions_to_file()

        now = time.time()
        if now - self.last_cache_update_ts > self.cache_update_interval:
            self._save_marker_cache()
            self.last_cache_update_ts = now

        # TODO anything needed?
        # while not self.cache_queue.empty():
        #     idx, c_m = self.cache_queue.get()
        #     self.cache.update(idx, c_m)
        #
        #     for s in self.surfaces:
        #         s.update_cache(self.cache, min_marker_perimeter=self.min_marker_perimeter,
        #                        min_id_confidence=self.min_id_confidence, idx=idx)
        #     if self.cacher_run.value is False:
        #         self.recalculate()
        #     if self.timeline:
        #         self.timeline.refresh()

    def _update_surface_locations(self, idx):
        for surface in self.surfaces:
            surface.update_location(idx, self.marker_cache, self.camera_model)

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

    def _update_surface_heatmap(self, surface):
        in_mark = self.g_pool.seek_control.trim_left
        out_mark = self.g_pool.seek_control.trim_right
        section = slice(in_mark, out_mark)

        all_gaze_timestamps = self.g_pool.timestamps
        all_gaze_positions = self.g_pool.gaze_positions

        self._compute_across_surfaces_heatmap(
            section, all_gaze_timestamps, all_gaze_positions
        )

        surface.update_heatmap(
            section, all_gaze_timestamps, all_gaze_positions, self.camera_model
        )

    def add_surface(self, _, init_dict=None):
        super().add_surface(_, init_dict=init_dict)
        # Plugin initialization loads surface definitions before UI is initialized. Changing timeline height will fail in this case.
        if self.markers or init_dict is not None:
            try:
                self.timeline.content_height += self.timeline_line_height
            except AttributeError:
                pass
        self.surfaces[-1].on_surface_changed = self.on_surface_change
        self._update_surface_heatmap(self.surfaces[-1])

    def remove_surface(self, _):
        super().remove_surface(_)
        self.timeline.content_height -= self.timeline_line_height

    def gl_display(self):
        if self.timeline:
            self.timeline.refresh()
        super().gl_display()
        # if self.mode == "Show Metrics":
        #     #todo: draw a backdrop to represent the gaze that is not on any surface
        #     for s in self.surfaces:
        #         #draw a quad on surface with false color of value.
        #         s.gl_display_metrics()

    def gl_display_cache_bars(self, width, height, scale):
        TS = self.g_pool.timestamps
        with gl_utils.Coord_System(TS[0], TS[-1], height, 0):
            # Lines for areas that have been cached
            cached_ranges = []
            for r in self.marker_cache.visited_ranges:  # [[0,1],[3,4]]
                cached_ranges += (
                    (TS[r[0]], 0),
                    (TS[r[1]], 0),
                )  # [(0,0),(1,0),(3,0),(4,0)]

            gl.glTranslatef(0, scale * self.timeline_line_height / 2, 0)
            color = pyglui_utils.RGBA(.8, .6, .2, .8)
            pyglui_utils.draw_polyline(
                cached_ranges, color=color, line_type=gl.GL_LINES, thickness=scale * 4
            )

            # Lines where surfaces have been found in video
            cached_surfaces = []
            for surface in self.surfaces:
                found_at = []
                if surface.location_cache is not None:
                    for r in surface.location_cache.positive_ranges:  # [[0,1],[3,4]]
                        found_at += (
                            (TS[r[0]], 0),
                            (TS[r[1]], 0),
                        )  # [(0,0),(1,0),(3,0),(4,0)]
                    cached_surfaces.append(found_at)

            color = pyglui_utils.RGBA(0, .7, .3, .8)

            for surface in cached_surfaces:
                gl.glTranslatef(0, scale * self.timeline_line_height, 0)
                pyglui_utils.draw_polyline(
                    surface, color=color, line_type=gl.GL_LINES, thickness=scale * 2
                )

    def draw_labels(self, width, height, scale):
        self.glfont.set_size(self.timeline_line_height * .8 * scale)
        self.glfont.draw_text(width, 0, "Marker Cache")
        for idx, s in enumerate(self.surfaces):
            gl.glTranslatef(0, self.timeline_line_height * scale, 0)
            self.glfont.draw_text(width, 0, s.name)

    def on_notify(self, notification):
        super().on_notify(notification)

        if notification["subject"] == "surface_tracker.marker_detection_params_changed":
            self.recalculate_marker_cache()
        elif notification["subject"] == "surface_tracker.marker_min_perimeter_changed":
            self._update_filtered_markers()
            for surface in self.surfaces:
                surface.cache = None
        elif notification["subject"].startswith(
            "surface_tracker.heatmap_params_changed"
        ):
            for surface in self.surfaces:
                if surface.uid == notification["uid"]:
                    self._update_surface_heatmap(surface)
                    break
        elif notification["subject"].startswith("seek_control.trim_indeces_changed"):
            for surface in self.surfaces:
                self._update_surface_heatmap(surface)
        elif notification["subject"] == "surface_tracker.surfaces_changed":
            for surface in self.surfaces:
                if surface.uid == notification["uid"]:
                    surface.location_cache = None
                    break

    def on_surface_change(self, surface):
        self.save_surface_definitions_to_file()
        self._update_surface_heatmap(surface)

    def deinit_ui(self):
        super().deinit_ui()
        self.g_pool.user_timelines.remove(self.timeline)
        self.timeline = None
        self.glfont = None

    def cleanup(self):
        super().cleanup()

        self._save_marker_cache()

    def _save_marker_cache(self):
        marker_cache_file = file_methods.Persistent_Dict(
            os.path.join(self.g_pool.rec_dir, "square_marker_cache")
        )
        marker_cache_file["marker_cache_unfiltered"] = list(
            self.marker_cache_unfiltered
        )
        marker_cache_file["version"] = self.marker_cache_version
        marker_cache_file["inverted_markers"] = self.inverted_markers
        marker_cache_file.save()
