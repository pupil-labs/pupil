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
import csv
import itertools

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
import player_methods

from surface_tracker.surface_tracker import Surface_Tracker
from surface_tracker import (
    offline_utils,
    background_tasks,
    Square_Marker_Detection,
    Heatmap_Mode,
)
from surface_tracker.surface_offline import Surface_Offline


# TODO Improve all docstrings, make methods privat appropriately
# TODO Two colors in timeline to indicate detected markrs vs frames without markers


class Surface_Tracker_Offline(Surface_Tracker, Analysis_Plugin_Base):
    """
    - Mostly extends the Surface Tracker with a cache
    Special version of surface tracker for use with videofile source.
    It uses a seperate process to search all frames in the world video file for markers.
     - self.cache is a list containing marker positions for each frame.
     - self.surfaces[i].cache is a list containing surface positions for each frame
    Both caches are build up over time. The marker cache is also session persistent.
    See marker_tracker.py for more info on this marker tracker.
    """

    # TODO Implement freeze feature
    # TODO test opening old recordings/surface definitions with new version
    # TODO recompute gaze on gaze change notification
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
        # Also add very small detected markers to cache and filter cache afterwards
        self.cache_min_marker_perimeter = 20
        self.cache_seek_idx = mp.Value("i", 0)
        self._init_marker_cache()
        self.last_cache_update_ts = time.time()
        self.cache_update_interval = 5

        self.gaze_on_surf_buffer = None
        self.gaze_on_surf_buffer_filler = None
        self.fixations_on_surf_buffer = None
        self.fixations_on_surf_buffer_filler = None

        self._heatmap_update_requests = set()
        self.make_export = False
        self.export_params = None

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
                    markers = [
                        Square_Marker_Detection(*args) if args else None
                        for args in markers
                    ]
                    self.marker_cache_unfiltered.append(markers)
                else:
                    self.marker_cache_unfiltered.append(markers)
            marker_cache_unfiltered = Cache_List(
                self.marker_cache_unfiltered, positive_eval_fn=_cache_pos_eval_fn
            )
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
                surface.location_cache = None

        self.marker_cache_unfiltered = Cache_List(
            previous_state, positive_eval_fn=_cache_pos_eval_fn
        )
        self._update_filtered_markers()

        self.cache_filler = background_tasks.background_video_processor(
            self.g_pool.capture.source_path,
            offline_utils.marker_detection_callable(
                self.cache_min_marker_perimeter, self.inverted_markers
            ),
            list(self.marker_cache),
            self.cache_seek_idx,
        )

    def recent_events(self, events):
        super().recent_events(events)

        if self.gaze_on_surf_buffer_filler is not None:
            for gof in self.gaze_on_surf_buffer_filler.fetch():
                try:
                    self.gaze_on_surf_buffer.append(gof)
                except AttributeError:
                    self.gaze_on_surf_buffer = []
                    self.gaze_on_surf_buffer.append(gof)

            # fixations will be gathered additionally to gaze if we want to make an export
            if self.fixations_on_surf_buffer_filler is not None:
                for gof in self.fixations_on_surf_buffer_filler.fetch():
                    try:
                        self.fixations_on_surf_buffer.append(gof)
                    except AttributeError:
                        self.fixations_on_surf_buffer = []
                        self.fixations_on_surf_buffer.append(gof)

            # Once all background processes are completed, update and export!
            if self.gaze_on_surf_buffer_filler.completed and (
                self.fixations_on_surf_buffer_filler is None
                or self.fixations_on_surf_buffer_filler.completed
            ):
                self.gaze_on_surf_buffer_filler = None
                self.fixations_on_surf_buffer_filler = None
                self._update_surface_heatmaps()
                if self.make_export:
                    self.save_surface_statsics_to_file()
                self.gaze_on_surf_buffer = None
                self.fixations_on_surf_buffer = None

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
        self.marker_cache = Cache_List(
            marker_cache, positive_eval_fn=_cache_pos_eval_fn
        )

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
        def set_name(val):

            names = [x.name for x in self.surfaces]
            if val in names and val != surface.name:
                logger.warning("The name '{}' is already in use!".format(val))
                return

            self.notify_all(
                {
                    "subject": "surface_tracker.surface_name_changed",
                    "old_name": surface.name,
                    "new_name": val,
                }
            )
            surface.name = val

        def set_x(val):
            if val <= 0:
                logger.warning("Surface size must be positive!")
                return

            surface.real_world_size["x"] = val
            self.notify_all(
                {
                    "subject": "surface_tracker.heatmap_params_changed",
                    "name": surface.name,
                }
            )

        def set_y(val):
            if val <= 0:
                logger.warning("Surface size must be positive!")
                return
            surface.real_world_size["y"] = val
            self.notify_all(
                {
                    "subject": "surface_tracker.heatmap_params_changed",
                    "name": surface.name,
                }
            )

        def set_hm_smooth(val):
            if val < 1:
                logger.warning("Heatmap Smoothness must be in (1,200)!")
                return
            surface._heatmap_scale_inv = val
            surface.heatmap_scale = 201 - val
            self.notify_all(
                {
                    "subject": "surface_tracker.heatmap_params_changed",
                    "name": surface.name,
                    "delay": 0.5,
                }
            )

        idx = self.surfaces.index(surface)
        s_menu = pyglui.ui.Growing_Menu("{}".format(self.surfaces[idx].name))
        s_menu.collapsed = True
        s_menu.append(pyglui.ui.Text_Input("name", surface, setter=set_name))
        s_menu.append(
            pyglui.ui.Text_Input(
                "x", surface.real_world_size, label="Width", setter=set_x
            )
        )
        s_menu.append(
            pyglui.ui.Text_Input(
                "y", surface.real_world_size, label="Height", setter=set_y
            )
        )
        s_menu.append(
            pyglui.ui.Slider(
                "_heatmap_scale_inv",
                surface,
                label="Heatmap Smoothness",
                setter=set_hm_smooth,
                step=1,
                min=1,
                max=200,
            )
        )
        s_menu.append(
            pyglui.ui.Button(
                "Open Surface in Window",
                self.gui.surface_windows[surface].open_close_window,
            )
        )
        remove_surf = lambda: self.remove_surface(idx)
        s_menu.append(pyglui.ui.Button("remove", remove_surf))
        self.menu.append(s_menu)

    def _compute_across_surfaces_heatmap(self):

        if self.gaze_on_surf_buffer is None:
            gazes_all = []
        else:
            gazes_all = list(itertools.chain.from_iterable(self.gaze_on_surf_buffer))

        gazes_on_surf = []
        for gof in gazes_all:
            gof = [g for g in gof if g["on_surf"]]
            gazes_on_surf.append(len(gof))

        if gazes_on_surf:
            max_res = max(gazes_on_surf)
            results = np.array(gazes_on_surf, dtype=np.float32)
            if max_res > 0:
                results *= 255. / max_res
            results = np.uint8(results)
            results_c_maps = cv2.applyColorMap(results, cv2.COLORMAP_JET)

            for s, c_map in zip(self.surfaces, results_c_maps):
                heatmap = np.ones((1, 1, 4), dtype=np.uint8) * 125
                heatmap[:, :, :3] = c_map
                s.across_surface_heatmap = heatmap
        else:
            for s in self.surfaces:
                s.across_surface_heatmap = s._get_dummy_heatmap()

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
            for surf in self.surfaces:
                self._heatmap_update_requests.add(surf)
            self.fill_gaze_on_surf_buffer()
            self._save_marker_cache()
            self.save_surface_definitions_to_file()

        now = time.time()
        if now - self.last_cache_update_ts > self.cache_update_interval:
            self._save_marker_cache()
            self.last_cache_update_ts = now

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

    def _update_surface_heatmaps(self):
        self._compute_across_surfaces_heatmap()

        for surface in self._heatmap_update_requests:
            surf_idx = self.surfaces.index(surface)
            surface.update_heatmap(self.gaze_on_surf_buffer[surf_idx])

        self._heatmap_update_requests = set()

    def fill_gaze_on_surf_buffer(self):
        if self.gaze_on_surf_buffer_filler is not None:
            self.gaze_on_surf_buffer_filler.cancel()

        in_mark = self.g_pool.seek_control.trim_left
        out_mark = self.g_pool.seek_control.trim_right
        section = slice(in_mark, out_mark)

        all_world_timestamps = self.g_pool.timestamps
        all_gaze_events = self.g_pool.gaze_positions

        self.gaze_on_surf_buffer_filler = background_tasks.background_gaze_on_surface(
            self.surfaces,
            section,
            all_world_timestamps,
            all_gaze_events,
            self.camera_model,
        )

        if self.make_export:
            if self.fixations_on_surf_buffer_filler is not None:
                self.fixations_on_surf_buffer_filler.cancel()

            all_fixation_events = self.g_pool.fixations

            self.fixations_on_surf_buffer_filler = background_tasks.background_gaze_on_surface(
                self.surfaces,
                section,
                all_world_timestamps,
                all_fixation_events,
                self.camera_model,
            )

    def add_surface(self, init_dict=None):
        super().add_surface(init_dict=init_dict)
        # Plugin initialization loads surface definitions before UI is initialized. Changing timeline height will fail in this case.
        if self.markers or init_dict is not None:
            try:
                self.timeline.content_height += self.timeline_line_height
                self.fill_gaze_on_surf_buffer()
            except AttributeError:
                pass
        self.surfaces[-1].on_surface_changed = self.on_surface_change

    def remove_surface(self, _):
        super().remove_surface(_)
        self.timeline.content_height -= self.timeline_line_height

    def gl_display(self):
        if self.timeline:
            self.timeline.refresh()
        super().gl_display()

    def gl_display_cache_bars(self, width, height, scale):
        TS = self.g_pool.timestamps
        with gl_utils.Coord_System(TS[0], TS[-1], height, 0):
            # Lines for areas that have been cached
            cached_ranges = []
            for r in self.marker_cache.visited_ranges:
                cached_ranges += ((TS[r[0]], 0), (TS[r[1]], 0))

            gl.glTranslatef(0, scale * self.timeline_line_height / 2, 0)
            color = pyglui_utils.RGBA(.8, .2, .2, .8)
            pyglui_utils.draw_polyline(
                cached_ranges, color=color, line_type=gl.GL_LINES, thickness=scale * 4
            )
            cached_ranges = []
            for r in self.marker_cache.positive_ranges:
                cached_ranges += ((TS[r[0]], 0), (TS[r[1]], 0))

            color = pyglui_utils.RGBA(0, .7, .3, .8)
            pyglui_utils.draw_polyline(
                cached_ranges, color=color, line_type=gl.GL_LINES, thickness=scale * 4
            )

            # Lines where surfaces have been found in video
            cached_surfaces = []
            for surface in self.surfaces:
                found_at = []
                if surface.location_cache is not None:
                    for r in surface.location_cache.positive_ranges:  # [[0,1],[3,4]]
                        found_at += ((TS[r[0]], 0), (TS[r[1]], 0))
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
                surface.location_cache = None

        elif notification["subject"] == "surface_tracker.heatmap_params_changed":
            for surface in self.surfaces:
                if surface.name == notification["name"]:
                    self._heatmap_update_requests.add(surface)
                    surface.within_surface_heatmap = surface._get_dummy_heatmap()
                    break
            self.fill_gaze_on_surf_buffer()

        elif notification["subject"].startswith("seek_control.trim_indices_changed"):
            for surface in self.surfaces:
                surface.within_surface_heatmap = surface._get_dummy_heatmap()
                self._heatmap_update_requests.add(surface)
            self.fill_gaze_on_surf_buffer()

        elif notification["subject"] == "surface_tracker.surfaces_changed":
            for surface in self.surfaces:
                if surface.name == notification["name"]:
                    surface.location_cache = None
                    surface.within_surface_heatmap = surface._get_dummy_heatmap()
                    self._heatmap_update_requests.add(surface)
                    break
            self.fill_gaze_on_surf_buffer()

        elif notification["subject"] == "should_export":
            self.make_export = True
            self.export_params = (notification["range"], notification["export_dir"])
            self.fill_gaze_on_surf_buffer()

    def on_surface_change(self, surface):
        self.save_surface_definitions_to_file()
        self._heatmap_update_requests.add(surface)
        self.fill_gaze_on_surf_buffer()

    # TODO dow we want to export dist_img transformations as well?
    # Make naming in raw data exporter consistent with this exporter
    def save_surface_statsics_to_file(self):
        """
        between in and out mark

            report: gaze distribution:
                    - total gazepoints
                    - gaze points on surface x
                    - gaze points not on any surface

            report: surface visisbility

                - total frames
                - surface x visible framecount

            surface events:
                frame_no, ts, surface "name", "id" enter/exit

            for each surface:
                fixations_on_name.csv
                gaze_on_name_id.csv
                positions_of_name_id.csv

        """
        export_range, export_dir = self.export_params
        metrics_dir = os.path.join(export_dir, "surfaces")
        section = slice(*export_range)
        in_mark = section.start
        out_mark = section.stop
        logger.info("exporting metrics to {}".format(metrics_dir))
        if os.path.isdir(metrics_dir):
            logger.info("Will overwrite previous export for this section")
        else:
            try:
                os.mkdir(metrics_dir)
            except:
                logger.warning("Could not make metrics dir {}".format(metrics_dir))
                return

        with open(
            os.path.join(metrics_dir, "surface_visibility.csv"),
            "w",
            encoding="utf-8",
            newline="",
        ) as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=",")

            # surface visibility report
            frame_count = len(self.g_pool.timestamps[section])

            csv_writer.writerow(("frame_count", frame_count))
            csv_writer.writerow((""))
            csv_writer.writerow(("surface_name", "visible_frame_count"))
            for surface in self.surfaces:
                if surface.location_cache == None:
                    logger.warning(
                        "The surface is not cached. Please wait for the cacher to collect data."
                    )
                    return
                visible_count = surface.visible_count_in_section(section)
                csv_writer.writerow((surface.name, visible_count))
            logger.info("Created 'surface_visibility.csv' file")

        with open(
            os.path.join(metrics_dir, "surface_gaze_distribution.csv"),
            "w",
            encoding="utf-8",
            newline="",
        ) as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=",")

            # gaze distribution report
            export_window = player_methods.exact_window(
                self.g_pool.timestamps, export_range
            )
            gaze_in_section = self.g_pool.gaze_positions.by_ts_window(export_window)
            not_on_any_surf_ts = set([gp["timestamp"] for gp in gaze_in_section])

            csv_writer.writerow(("total_gaze_point_count", len(gaze_in_section)))
            csv_writer.writerow((""))
            csv_writer.writerow(("surface_name", "gaze_count"))

            for surf_idx, surface in enumerate(self.surfaces):
                gaze_on_surf = self.gaze_on_surf_buffer[surf_idx]
                gaze_on_surf = list(itertools.chain.from_iterable(gaze_on_surf))
                gaze_on_surf_ts = set(
                    [gp["base_data"][1] for gp in gaze_on_surf if gp["on_surf"]]
                )
                not_on_any_surf_ts -= gaze_on_surf_ts
                csv_writer.writerow((surface.name, len(gaze_on_surf_ts)))

            csv_writer.writerow(("not_on_any_surface", len(not_on_any_surf_ts)))
            logger.info("Created 'surface_gaze_distribution.csv' file")

        with open(
            os.path.join(metrics_dir, "surface_events.csv"),
            "w",
            encoding="utf-8",
            newline="",
        ) as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=",")

            # surface events report
            csv_writer.writerow(
                ("world_idx", "world_timestamp", "surface_name", "event_type")
            )

            events = []
            for surface in self.surfaces:
                for (
                    enter_frame_id,
                    exit_frame_id,
                ) in surface.location_cache.positive_ranges:
                    events.append(
                        {
                            "frame_id": enter_frame_id,
                            "surf_name": surface.name,
                            "event": "enter",
                        }
                    )
                    events.append(
                        {
                            "frame_id": exit_frame_id,
                            "surf_name": surface.name,
                            "event": "exit",
                        }
                    )

            events.sort(key=lambda x: x["frame_id"])
            for e in events:
                csv_writer.writerow(
                    (
                        e["frame_id"],
                        self.g_pool.timestamps[e["frame_id"]],
                        e["surf_name"],
                        e["event"],
                    )
                )
            logger.info("Created 'surface_events.csv' file")

        for surf_idx, surface in enumerate(self.surfaces):
            # per surface names:
            surface_name = "_" + surface.name.replace("/", "")

            # save surface_positions as csv
            with open(
                os.path.join(metrics_dir, "surf_positons" + surface_name + ".csv"),
                "w",
                encoding="utf-8",
                newline="",
            ) as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=",")
                csv_writer.writerow(
                    (
                        "world_idx",
                        "world_timestamp",
                        "img_to_surf_trans",
                        "surf_to_img_trans",
                        "num_detected_markers",
                    )
                )
                for idx, ts, ref_surf_data in zip(
                    range(len(self.g_pool.timestamps)),
                    self.g_pool.timestamps,
                    surface.location_cache,
                ):
                    if in_mark <= idx < out_mark:
                        if (
                            ref_surf_data is not None
                            and ref_surf_data is not False
                            and ref_surf_data["detected"]
                        ):
                            csv_writer.writerow(
                                (
                                    idx,
                                    ts,
                                    ref_surf_data["img_to_surf_trans"],
                                    ref_surf_data["surf_to_img_trans"],
                                    ref_surf_data["num_detected_markers"],
                                )
                            )

            # save gaze on surf as csv.
            with open(
                os.path.join(
                    metrics_dir, "gaze_positions_on_surface" + surface_name + ".csv"
                ),
                "w",
                encoding="utf-8",
                newline="",
            ) as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=",")
                csv_writer.writerow(
                    (
                        "world_timestamp",
                        "world_idx",
                        "gaze_timestamp",
                        "x_norm",
                        "y_norm",
                        "x_scaled",
                        "y_scaled",
                        "on_surf",
                        "confidence",
                    )
                )
                for idx, gaze_on_surf in enumerate(self.gaze_on_surf_buffer[surf_idx]):
                    idx += in_mark
                    if gaze_on_surf:
                        for gp in gaze_on_surf:
                            csv_writer.writerow(
                                (
                                    self.g_pool.timestamps[idx],
                                    idx,
                                    gp["timestamp"],
                                    gp["norm_pos"][0],
                                    gp["norm_pos"][1],
                                    gp["norm_pos"][0] * surface.real_world_size["x"],
                                    gp["norm_pos"][1] * surface.real_world_size["y"],
                                    gp["on_surf"],
                                    gp["confidence"],
                                )
                            )
            # save fixations on surf as csv.
            with open(
                os.path.join(
                    metrics_dir, "fixations_on_surface" + surface_name + ".csv"
                ),
                "w",
                encoding="utf-8",
                newline="",
            ) as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=",")
                csv_writer.writerow(
                    (
                        "start_timestamp",
                        "norm_pos_x",
                        "norm_pos_y",
                        "x_scaled",
                        "y_scaled",
                        "on_surf",
                    )
                )
                for idx, fix_on_surf in enumerate(
                    self.fixations_on_surf_buffer[surf_idx]
                ):
                    idx += in_mark
                    if fix_on_surf:
                        without_duplicates = dict(
                            [(fix["base_data"][1], fix) for fix in fix_on_surf]
                        ).values()
                        for fix in without_duplicates:
                            csv_writer.writerow(
                                (
                                    self.g_pool.timestamps[idx],
                                    idx,
                                    fix["timestamp"],
                                    fix["norm_pos"][0],
                                    fix["norm_pos"][1],
                                    fix["norm_pos"][0] * surface.real_world_size["x"],
                                    fix["norm_pos"][1] * surface.real_world_size["y"],
                                    fix["on_surf"],
                                    fix["confidence"],
                                )
                            )

            logger.info(
                "Saved surface positon gaze and fixation data for '{}'".format(
                    surface.name
                )
            )

            if surface.within_surface_heatmap is not None:
                logger.info("Saved Heatmap as .png file.")
                cv2.imwrite(
                    os.path.join(metrics_dir, "heatmap" + surface_name + ".png"),
                    surface.within_surface_heatmap,
                )

        logger.info("Done exporting reference surface data.")
        # TODO enable export of surface image?
        # if s.detected and self.img is not None:
        #     #let save out the current surface image found in video

        #     #here we get the verts of the surface quad in norm_coords
        #     mapped_space_one = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32).reshape(-1,1,2)
        #     screen_space = cv2.perspectiveTransform(mapped_space_one,s.m_to_screen).reshape(-1,2)
        #     #now we convert to image pixel coods
        #     screen_space[:,1] = 1-screen_space[:,1]
        #     screen_space[:,1] *= self.img.shape[0]
        #     screen_space[:,0] *= self.img.shape[1]
        #     s_0,s_1 = s.real_world_size
        #     #no we need to flip vertically again by setting the mapped_space verts accordingly.
        #     mapped_space_scaled = np.array(((0,s_1),(s_0,s_1),(s_0,0),(0,0)),dtype=np.float32)
        #     M = cv2.getPerspectiveTransform(screen_space,mapped_space_scaled)
        #     #here we do the actual perspactive transform of the image.
        #     surf_in_video = cv2.warpPerspective(self.img,M, (int(s.real_world_size['x']),int(s.real_world_size['y'])) )
        #     cv2.imwrite(os.path.join(metrics_dir,'surface'+surface_name+'.png'),surf_in_video)
        #     logger.info("Saved current image as .png file.")
        # else:
        #     logger.info("'%s' is not currently visible. Seek to appropriate frame and repeat this command."%s.name)

        self.make_export = False

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


_cache_pos_eval_fn = lambda x: (x is not False) and len(x) > 0
