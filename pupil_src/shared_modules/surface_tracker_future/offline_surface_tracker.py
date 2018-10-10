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

logger = logging.getLogger(__name__)
import platform
import multiprocessing

if platform.system() == "Darwin":
    mp = multiprocessing.get_context("fork")
else:
    mp = multiprocessing.get_context()
import time

import pyglui
import gl_utils
import pyglui.cygl.utils as pyglui_utils
import OpenGL.GL as gl

from plugin import Analysis_Plugin_Base
import file_methods
from cache_list import Cache_List

from .surface_tracker import Surface_Tracker_Future, Marker
from . import offline_utils
from . import gui
from . import background_tasks
from .offline_surface import Offline_Surface


class Offline_Surface_Tracker_Future(Surface_Tracker_Future, Analysis_Plugin_Base):
    """
    # TODO if you add a surface while no markers are visible, the next frame containing markers will be used
    # TODO Improve docstring
    - Mostly extends the Surface Tracker with a cache
    Special version of surface tracker for use with videofile source.
    It uses a seperate process to search all frames in the world video file for markers.
     - self.cache is a list containing marker positions for each frame.
     - self.surfaces[i].cache is a list containing surface positions for each frame
    Both caches are build up over time. The marker cache is also session persistent.
    See marker_tracker.py for more info on this marker tracker.
    """

    # TODO make surfaces are saved on creation
    # TODO make sure square_marker_cache is saved continuously during creation.
    # TODO Make methods private appropriatly
    def __init__(
        self,
        g_pool,
        marker_min_perimeter=60,
        marker_min_confidence=0.6,
        inverted_markers=False,
    ):
        self.timeline_line_height = 16
        super().__init__(
            g_pool, marker_min_perimeter, marker_min_confidence, inverted_markers
        )
        self.order = .2

        # Caches
        self.marker_cache_version = 3
        self.cache_min_marker_perimeter = 20  # find even super small markers. The
        # surface locater will filter using min_marker_perimeter
        self.cache_seek_idx = mp.Value("i", 0)
        self._init_marker_cache()
        self.last_cache_update_ts = time.time()
        self.cache_update_interval = 2
        # self.recalculate() # What does this do?

    @property
    def Surface_Class(self):
        return Offline_Surface

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
        # TODO recompute marker_cache when filtering parameters change
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
        self.add_menu()
        self.menu.label = "Offline Surface Tracker"
        self.add_button = pyglui.ui.Thumb(
            "add_surface",
            setter=lambda x: self.add_surface(),
            getter=lambda: False,
            label="A",
            hotkey="a",
        )
        self.g_pool.quickbar.append(self.add_button)

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

        self.update_ui()

    def update_ui(self):
        def set_marker_min_perimeter(val):
            self.marker_min_perimeter = val
            self.notify_all({"subject": "marker_filtering_params_changed", "delay": 1})

        def set_marker_min_confidence(val):
            self.marker_min_confidence = val
            self.notify_all({"subject": "marker_filtering_params_changed", "delay": 1})

        def set_invert_image(val):
            self.inverted_markers = val
            self.recalculate_marker_cache()

        def set_robust_detection(val):
            self.robust_detection = val
            self.recalculate_marker_cache()

        try:
            self.menu.elements[:] = []
        except AttributeError:
            return
        self.menu.append(
            pyglui.ui.Info_Text(
                "The offline surface tracker will look for markers in the entire video. By default it uses surfaces defined in capture. You can change and add more surfaces here."
            )
        )
        self.menu.append(
            pyglui.ui.Info_Text(
                "Press the export button or type 'e' to start the export."
            )
        )
        self.menu.append(
            pyglui.ui.Switch(
                "robust_detection",
                self,
                setter=set_robust_detection,
                label="Robust detection",
            )
        )  # TODO is robust_detection atually used?
        self.menu.append(
            pyglui.ui.Switch(
                "inverted_markers",
                self,
                setter=set_invert_image,
                label="Use inverted markers",
            )
        )
        self.menu.append(
            pyglui.ui.Slider(
                "marker_min_perimeter",
                self,
                setter=set_marker_min_perimeter,
                step=1,
                min=30,
                max=100,
            )
        )
        self.menu.append(
            pyglui.ui.Slider(
                "marker_min_confidence",
                self,
                setter=set_marker_min_confidence,
                step=0.01,
                min=0,
                max=1,
            )
        )
        self.menu.append(
            pyglui.ui.Selector(
                "state",
                self.gui,
                label="Mode",
                labels=[e.value for e in gui.State],
                selection=[e for e in gui.State],
            )
        )
        self.menu.append(
            pyglui.ui.Info_Text(
                'To see heatmap or surface metrics visualizations, click (re)-calculate gaze distributions. Set "X size" and "Y size" for each surface to see heatmap visualizations.'
            )
        )
        self.menu.append(
            pyglui.ui.Button("(Re)-calculate gaze distributions", self.recalculate)
        )
        self.menu.append(pyglui.ui.Button("Add surface", lambda: self.add_surface("_")))

        for s in self.surfaces:
            idx = self.surfaces.index(s)
            s_menu = pyglui.ui.Growing_Menu("Surface {}".format(idx))
            s_menu.collapsed = True
            s_menu.append(pyglui.ui.Text_Input("name", s))
            s_menu.append(pyglui.ui.Text_Input("x", s.real_world_size, label="X size"))
            s_menu.append(pyglui.ui.Text_Input("y", s.real_world_size, label="Y size"))
            s_menu.append(
                pyglui.ui.Text_Input(
                    "gaze_history_length", s, label="Gaze History Length [seconds]"
                )
            )
            s_menu.append(
                pyglui.ui.Button(
                    "Open Debug Window", self.gui.surface_windows[s].open_close_window
                )
            )

            def make_remove_s(i):
                return lambda: self.remove_surface(i)

            remove_s = make_remove_s(idx)
            s_menu.append(pyglui.ui.Button("remove", remove_s))
            self.menu.append(s_menu)

    def load_surface_definitions_from_file(self):
        surface_definitions = file_methods.Persistent_Dict(
            os.path.join(self.g_pool.rec_dir, "surface_definitions")
        )

        for init_dict in surface_definitions.get("surfaces", []):
            self.add_surface(None, init_dict=init_dict)

    def recalculate(self):
        pass  # TODO implement
        # in_mark = self.g_pool.seek_control.trim_left
        # out_mark = self.g_pool.seek_control.trim_right
        # section = slice(in_mark,out_mark)
        #
        # # calc heatmaps
        # for s in self.surfaces:
        #     if s.defined:
        #         s.generate_heatmap(section)
        #
        # # calc distirbution accross all surfaces.
        # results = []
        # for s in self.surfaces:
        #     gaze_on_surf  = s.gaze_on_surf_in_section(section)
        #     results.append(len(gaze_on_surf))
        #     self.metrics_gazecount = len(gaze_on_surf)
        #
        # if results == []:
        #     logger.warning("No surfaces defined.")
        #     return
        # max_res = max(results)
        # results = np.array(results,dtype=np.float32)
        # if not max_res:
        #     logger.warning("No gaze on any surface for this section!")
        # else:
        #     results *= 255./max_res
        # results = np.uint8(results)
        # results_c_maps = cv2.applyColorMap(results, cv2.COLORMAP_JET)
        #
        # for s,c_map in zip(self.surfaces,results_c_maps):
        #     heatmap = np.ones((1,1,4),dtype=np.uint8)*125
        #     heatmap[:,:,:3] = c_map
        #     s.metrics_texture = Named_Texture()
        #     s.metrics_texture.update_from_ndarray(heatmap)

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return

        self.current_frame_idx = frame.index

        if not self.cache_filler is None:
            self._update_marker_and_surface_caches()

        # Move seek index to current frame if caches do not contain data for it
        self.markers = self.marker_cache[frame.index]
        if self.markers is False:
            self.markers = []
            self.cache_seek_idx.value = frame.index

        self._update_surfaces(frame.index)
        self._surface_interactions(events, frame)

    def _update_marker_and_surface_caches(self):
        for idx, markers in self.cache_filler.fetch():
            self.marker_cache_unfiltered.update(idx, markers)
            markers_filtered = self._filter_markers(markers)
            self.marker_cache.update(idx, markers_filtered)

            for surf in self.surfaces:
                surf.update_cache(idx, self.marker_cache, self.camera_model)

        if self.cache_filler.completed:
            self.cache_filler = None

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

    def _update_surfaces(self, idx):
        for surface in self.surfaces:
            surface.update_location(idx, self.marker_cache, self.camera_model)

    def _move_surface_corners(self):
        for surface, corner_idx in self._edit_surf_verts:
            if surface.detected:
                surface.move_corner(
                    self.current_frame_idx,
                    self.marker_cache,
                    corner_idx,
                    self._last_mouse_pos.copy(),
                    self.camera_model,
                )

    def add_surface(self, _, init_dict=None):
        super().add_surface(_, init_dict=init_dict)
        # Plugin initialization loads surface definitions before UI is initialized. Changing timeline height will fail in this case.
        if self.markers or init_dict is not None:
            try:
                self.timeline.content_height += self.timeline_line_height
            except AttributeError:
                pass

    def remove_surface(self, _):
        super().remove_surface(_)
        self.timeline.content_height -= self.timeline_line_height

    # TODO does it make sense to move timeline stuff into gui class?
    def gl_display(self):
        if self.timeline:
            self.timeline.refresh()
        super().gl_display()
        # if self.mode == "Show Metrics":
        #     #todo: draw a backdrop to represent the gaze that is not on any surface
        #     for s in self.surfaces:
        #         #draw a quad on surface with false color of value.
        #         s.gl_display_metrics() # todo add support for metrics

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
            for s in self.surfaces:
                found_at = []
                if s.cache is not None:
                    for r in s.cache.positive_ranges:  # [[0,1],[3,4]]
                        found_at += (
                            (TS[r[0]], 0),
                            (TS[r[1]], 0),
                        )  # [(0,0),(1,0),(3,0),(4,0)]
                    cached_surfaces.append(found_at)

            color = pyglui_utils.RGBA(0, .7, .3, .8)

            for s in cached_surfaces:
                gl.glTranslatef(0, scale * self.timeline_line_height, 0)
                pyglui_utils.draw_polyline(
                    s, color=color, line_type=gl.GL_LINES, thickness=scale * 2
                )

    def on_notify(self, notification):
        super().on_notify(notification)
        if notification["subject"] == "marker_filtering_params_changed":
            self._update_filtered_markers()
            for surface in self.surfaces:
                surface.cache = None

    def draw_labels(self, width, height, scale):
        self.glfont.set_size(self.timeline_line_height * .8 * scale)
        self.glfont.draw_text(width, 0, "Marker Cache")
        for idx, s in enumerate(self.surfaces):
            gl.glTranslatef(0, self.timeline_line_height * scale, 0)
            self.glfont.draw_text(width, 0, s.name)

    def save_surface_definitions_to_file(self):
        logger.info("Save updated surfaces!")  # TODO remove
        surface_definitions = file_methods.Persistent_Dict(
            os.path.join(self.g_pool.rec_dir, "surface_definitions")
        )
        surface_definitions["surfaces"] = [
            surface.save_to_dict() for surface in self.surfaces if surface.defined
        ]
        surface_definitions.save()

    def deinit_ui(self):
        self.g_pool.user_timelines.remove(self.timeline)
        self.timeline = None
        self.glfont = None
        self.remove_menu()
        if self.add_button:
            self.g_pool.quickbar.remove(self.add_button)
            self.add_button = None

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
