"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import csv
import itertools
import logging
import os
import types

import background_helper
import cv2
import file_methods
import player_methods

from .surface_marker import Surface_Marker

logger = logging.getLogger(__name__)


def background_video_processor(
    video_file_path, callable, visited_list, seek_idx, mp_context
):
    return background_helper.IPC_Logging_Task_Proxy(
        "Background Video Processor",
        video_processing_generator,
        (video_file_path, callable, seek_idx, visited_list),
        context=mp_context,
    )


def video_processing_generator(video_file_path, callable, seek_idx, visited_list):
    import logging
    import os

    logger = logging.getLogger(__name__ + " with pid: " + str(os.getpid()))
    logger.debug("Started cacher process for Marker Detector")
    import video_capture

    cap = video_capture.File_Source(
        types.SimpleNamespace(),
        source_path=video_file_path,
        fill_gaps=True,
        timing=None,
    )

    # Ensure that indiced are not generated beyond video frame count
    frame_count = cap.get_frame_count()
    visited_list = visited_list[:frame_count]

    visited_list = [x is not None for x in visited_list]

    def next_unvisited_idx(frame_idx):
        """
        Starting from the given index, find the next frame that has not been
        processed yet. If no future frames need processing, check from the start.

        Args:
            frame_idx: Index to start search from.

        Returns: Next index that requires processing.

        """
        try:
            visited = visited_list[frame_idx]
        except IndexError:
            visited = True  # trigger search from the start

        if not visited:
            next_unvisited = frame_idx
        else:
            # find next unvisited site in the future
            try:
                next_unvisited = visited_list.index(False, frame_idx)
            except ValueError:
                # any thing in the past?
                try:
                    next_unvisited = visited_list.index(False, 0, frame_idx)
                except ValueError:
                    # no unvisited sites left. Done!
                    logger.debug("Caching completed.")
                    next_unvisited = None
        return next_unvisited

    def handle_frame(frame_idx):
        if frame_idx != cap.get_frame_index() + 1:
            # we need to seek:
            logger.debug(f"Seeking to Frame {frame_idx}")
            try:
                cap.seek_to_frame(frame_idx)
            except video_capture.FileSeekError:
                logger.warning(f"Could not evaluate frame: {frame_idx}.")
                visited_list[frame_idx] = True  # this frame is now visited.
                return []

        try:
            frame = cap.get_frame()
        except video_capture.EndofVideoError:
            logger.warning(f"Could not evaluate frame: {frame_idx}.")
            visited_list[frame_idx] = True
            return []
        return callable(frame)

    while True:
        last_frame_idx = cap.get_frame_index()
        if seek_idx.value != -1:
            assert seek_idx.value < len(
                visited_list
            ), "The requested seek index is outside of the predefined cache range!"
            last_frame_idx = seek_idx.value
            seek_idx.value = -1
            logger.debug(
                f"User required seek. Marker caching at Frame: {last_frame_idx}"
            )

        next_frame_idx = next_unvisited_idx(last_frame_idx)

        if next_frame_idx is None:
            break
        else:
            res = handle_frame(next_frame_idx)
            visited_list[next_frame_idx] = True
            yield next_frame_idx, res


def background_data_processor(data, callable, seek_idx, mp_context):
    return background_helper.IPC_Logging_Task_Proxy(
        "Background Data Processor",
        data_processing_generator,
        (data, callable, seek_idx),
        context=mp_context,
    )


def data_processing_generator(data, callable, seek_idx):
    # We treat frames without marker detections as already processed from the start.
    visited_list = [x is None for x in data]

    def next_unvisited_idx(sample_idx):
        """
        Starting from the given index, find the next sample that has not been
        processed yet. If no future samples need processing, check from the start.

        Args:
            sample_idx: Index to start search from.

        Returns: Next index that requires processing.

        """
        try:
            visited = visited_list[sample_idx]
        except IndexError:
            visited = True  # trigger search from the start

        if visited is False:
            next_unvisited = sample_idx
        else:
            # find next unvisited site in the future
            try:
                next_unvisited = visited_list.index(False, sample_idx)
            except ValueError:
                # any thing in the past?
                try:
                    next_unvisited = visited_list.index(False, 0, sample_idx)
                except ValueError:
                    next_unvisited = None
        return next_unvisited

    def handle_sample(sample_idx):
        sample = data[sample_idx]
        return callable(sample)

    next_sample_idx = 0
    while True:
        if seek_idx.value != -1:
            next_sample_idx = seek_idx.value
            seek_idx.value = -1

        next_sample_idx = next_unvisited_idx(next_sample_idx)

        if next_sample_idx is None:
            break
        else:
            res = handle_sample(next_sample_idx)
            visited_list[next_sample_idx] = True
            yield next_sample_idx, res
            next_sample_idx += 1


def gaze_on_surface_generator(
    surfaces, section, all_world_timestamps, all_gaze_events, camera_model
):
    for surface in surfaces:
        gaze_on_surf = surface.map_section(
            section, all_world_timestamps, all_gaze_events, camera_model
        )
        yield gaze_on_surf


def background_gaze_on_surface(
    surfaces, section, all_world_timestamps, all_gaze_events, camera_model, mp_context
):
    return background_helper.IPC_Logging_Task_Proxy(
        "Background Data Processor",
        gaze_on_surface_generator,
        (surfaces, section, all_world_timestamps, all_gaze_events, camera_model),
        context=mp_context,
    )


def get_export_proxy(
    export_dir,
    export_range,
    surfaces,
    world_timestamps,
    gaze_positions,
    fixations,
    camera_model,
    marker_cache_path,
    mp_context,
):
    exporter = Exporter(
        export_dir,
        export_range,
        surfaces,
        world_timestamps,
        gaze_positions,
        fixations,
        camera_model,
        marker_cache_path,
    )
    proxy = background_helper.IPC_Logging_Task_Proxy(
        "Offline Surface Tracker Exporter",
        exporter.save_surface_statisics_to_file,
        context=mp_context,
    )
    return proxy


class Exporter:
    def __init__(
        self,
        export_dir,
        export_range,
        surfaces,
        world_timestamps,
        gaze_positions,
        fixations,
        camera_model,
        marker_cache_path,
    ):
        self.export_range = export_range
        self.metrics_dir = os.path.join(export_dir, "surfaces")
        self.surfaces = surfaces
        self.world_timestamps = world_timestamps
        self.gaze_positions = gaze_positions
        self.fixations = fixations
        self.camera_model = camera_model
        self.gaze_on_surfaces = None
        self.fixations_on_surfaces = None
        self.marker_cache_path = marker_cache_path

    def save_surface_statisics_to_file(self):
        logger.info(f"exporting metrics to {self.metrics_dir}")
        if os.path.isdir(self.metrics_dir):
            logger.info("Will overwrite previous export for this section")
        else:
            try:
                os.mkdir(self.metrics_dir)
            except OSError:
                logger.warning(f"Could not make metrics dir {self.metrics_dir}")
                return

        (
            self.gaze_on_surfaces,
            self.fixations_on_surfaces,
        ) = self._map_gaze_and_fixations()

        self._export_surface_visibility()
        self._export_surface_gaze_distribution()
        self._export_surface_events()

        for surf_idx, surface in enumerate(self.surfaces):
            # Sanitize surface name to include it in the filename
            surface_name = "_" + surface.name.replace("/", "")

            self._export_surface_positions(surface, surface_name)
            self._export_gaze_on_surface(
                self.gaze_on_surfaces[surf_idx], surface, surface_name
            )
            self._export_fixations_on_surface(
                self.fixations_on_surfaces[surf_idx], surface, surface_name
            )
            self._export_surface_heatmap(surface, surface_name)

            logger.info(f"Saved surface gaze and fixation data for '{surface.name}'")

        # Cleanup surface related data to release memory
        self.surfaces = None
        self.fixations = None
        self.gaze_positions = None
        self.gaze_on_surfaces = None
        self.fixations_on_surfaces = None

        # Perform marker export *after* surface data is released
        # to avoid holding everything in memory all at once.
        self._export_marker_detections()

        logger.info("Done exporting reference surface data.")
        return

        # Task_proxy requires a genrator. The `yield` below
        # triggers this function to become a generator.
        yield

    def _map_gaze_and_fixations(self):
        """
        Result: Tuple[events_per_surface, events_per_surface]
        events_per_surface = List[events_surface_0, ..., events_surface_N]
        events_surface_i = List[events_world_frame_0, ..., events_world_frame_M]
        events_world_frame_j: List[event, ...]

        N: Number of surfaces
        M: Number of world frames
        """
        section = slice(*self.export_range)
        gaze_on_surface = list(
            gaze_on_surface_generator(
                self.surfaces,
                section,
                self.world_timestamps,
                self.gaze_positions,
                self.camera_model,
            )
        )

        fixations_on_surface = list(
            gaze_on_surface_generator(
                self.surfaces,
                section,
                self.world_timestamps,
                self.fixations,
                self.camera_model,
            )
        )

        return gaze_on_surface, fixations_on_surface

    def _export_marker_detections(self):
        export_slice = slice(*self.export_range)
        world_indices = range(*self.export_range)

        # Load the temporary marker cache created by the offline surface tracker
        marker_cache = file_methods.Persistent_Dict(self.marker_cache_path)
        marker_cache = marker_cache["marker_cache"][export_slice]

        incomplete_marker_cache_detected = False

        file_path = os.path.join(self.metrics_dir, "marker_detections.csv")

        try:
            with open(file_path, "w", encoding="utf-8", newline="") as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=",")
                csv_writer.writerow(
                    (
                        "world_index",
                        "marker_uid",
                        "corner_0_x",
                        "corner_0_y",
                        "corner_1_x",
                        "corner_1_y",
                        "corner_2_x",
                        "corner_2_y",
                        "corner_3_x",
                        "corner_3_y",
                    )
                )
                for world_index, serialized_markers in zip(world_indices, marker_cache):
                    if serialized_markers is None:
                        # set flag to break from outer loop
                        incomplete_marker_cache_detected = True

                    if incomplete_marker_cache_detected:
                        break  # outer loop

                    for sm in serialized_markers:
                        if sm is None:
                            # set flag to break from outer loop
                            incomplete_marker_cache_detected = True
                            break  # inner loop
                        m = Surface_Marker.deserialize(sm)
                        flat_corners = [x for c in m.verts_px for x in c[0]]
                        assert len(flat_corners) == 8  # sanity check
                        csv_writer.writerow(
                            (
                                world_index,
                                m.uid,
                                *flat_corners,
                            )
                        )
        finally:
            if incomplete_marker_cache_detected:
                logger.error("Marker detection not finished. No data will be exported.")
                # Delete incomplete marker cache export file
                os.remove(file_path)

            # Delete the temporary marker cache created by the offline surface tracker
            os.remove(self.marker_cache_path)
            self.marker_cache_path = None

    def _export_surface_visibility(self):
        with open(
            os.path.join(self.metrics_dir, "surface_visibility.csv"),
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")

            section = slice(*self.export_range)
            frame_count = len(self.world_timestamps[section])

            csv_writer.writerow(("frame_count", frame_count))
            csv_writer.writerow("")
            csv_writer.writerow(("surface_name", "visible_frame_count"))
            for surface in self.surfaces:
                if surface.location_cache is None:
                    logger.warning(
                        "The surface is not cached. Please wait for the cacher to "
                        "collect data."
                    )
                    return
                visible_count = surface.visible_count_in_section(section)
                csv_writer.writerow((surface.name, visible_count))
            logger.info("Created 'surface_visibility.csv' file")

    def _export_surface_gaze_distribution(self):
        with open(
            os.path.join(self.metrics_dir, "surface_gaze_distribution.csv"),
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")

            export_window = player_methods.exact_window(
                self.world_timestamps, self.export_range
            )
            gaze_in_section = self.gaze_positions.by_ts_window(export_window)
            not_on_any_surf_ts = {gp["timestamp"] for gp in gaze_in_section}

            csv_writer.writerow(("total_gaze_point_count", len(gaze_in_section)))
            csv_writer.writerow("")
            csv_writer.writerow(("surface_name", "gaze_count"))

            for surf_idx, surface in enumerate(self.surfaces):
                gaze_on_surf = self.gaze_on_surfaces[surf_idx]
                gaze_on_surf = list(itertools.chain.from_iterable(gaze_on_surf))
                gaze_on_surf_ts = {
                    gp["base_data"][1] for gp in gaze_on_surf if gp["on_surf"]
                }
                not_on_any_surf_ts -= gaze_on_surf_ts
                csv_writer.writerow((surface.name, len(gaze_on_surf_ts)))

            csv_writer.writerow(("not_on_any_surface", len(not_on_any_surf_ts)))
            logger.info("Created 'surface_gaze_distribution.csv' file")

    def _export_surface_events(self):
        with open(
            os.path.join(self.metrics_dir, "surface_events.csv"),
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")

            csv_writer.writerow(
                ("world_index", "world_timestamp", "surface_name", "event_type")
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
                        self.world_timestamps[e["frame_id"]],
                        e["surf_name"],
                        e["event"],
                    )
                )
            logger.info("Created 'surface_events.csv' file")

    def _export_surface_heatmap(self, surface, surface_name):
        if surface.within_surface_heatmap is not None:
            logger.info("Saved Heatmap as .png file.")
            heatmap_file_name = "heatmap" + surface_name + ".png"
            heatmap_path = os.path.join(self.metrics_dir, heatmap_file_name)

            surface_size = surface.real_world_size
            export_resolution = (surface_size["x"], surface_size["y"])

            MIN_EXPORT_DIMENSION = 2000  # min width and height in px
            scaling = MIN_EXPORT_DIMENSION / min(export_resolution)
            # NOTE: cv2.resize expects a tuple of ints specifically
            export_resolution = tuple(
                int(round(val * scaling)) for val in export_resolution
            )

            # throw away alpha channel for export
            heatmap_img = surface.within_surface_heatmap[:, :, :3]
            heatmap_img = cv2.resize(
                heatmap_img, export_resolution, interpolation=cv2.INTER_NEAREST
            )
            cv2.imwrite(heatmap_path, heatmap_img)

    def _export_surface_positions(self, surface, surface_name):
        with open(
            os.path.join(self.metrics_dir, "surf_positions" + surface_name + ".csv"),
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            csv_writer.writerow(
                (
                    "world_index",
                    "world_timestamp",
                    "img_to_surf_trans",
                    "surf_to_img_trans",
                    "num_detected_markers",
                    "dist_img_to_surf_trans",
                    "surf_to_dist_img_trans",
                    "num_definition_markers",
                )
            )
            surface_definition_marker_count = len(surface.registered_marker_uids)
            for idx, (ts, ref_surf_data) in enumerate(
                zip(self.world_timestamps, surface.location_cache)
            ):
                if self.export_range[0] <= idx < self.export_range[1]:
                    if (
                        ref_surf_data is not None
                        and ref_surf_data is not False
                        and ref_surf_data.detected
                    ):
                        csv_writer.writerow(
                            (
                                idx,
                                ts,
                                ref_surf_data.img_to_surf_trans,
                                ref_surf_data.surf_to_img_trans,
                                ref_surf_data.num_detected_markers,
                                ref_surf_data.dist_img_to_surf_trans,
                                ref_surf_data.surf_to_dist_img_trans,
                                surface_definition_marker_count,
                            )
                        )

    def _export_gaze_on_surface(self, gazes_on_surface, surface, surface_name):
        with open(
            os.path.join(
                self.metrics_dir, "gaze_positions_on_surface" + surface_name + ".csv"
            ),
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            csv_writer.writerow(
                (
                    "world_timestamp",
                    "world_index",
                    "gaze_timestamp",
                    "x_norm",
                    "y_norm",
                    "x_scaled",
                    "y_scaled",
                    "on_surf",
                    "confidence",
                )
            )
            for idx, gaze_on_surf in enumerate(gazes_on_surface):
                idx += self.export_range[0]
                if gaze_on_surf:
                    for gp in gaze_on_surf:
                        csv_writer.writerow(
                            (
                                self.world_timestamps[idx],
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

    def _export_fixations_on_surface(self, fixations_on_surf, surface, surface_name):
        """
        fixations_on_surf structure: fixations_on_surf[world_frame_idx][event_idx]
        """
        with open(
            os.path.join(
                self.metrics_dir, "fixations_on_surface" + surface_name + ".csv"
            ),
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            csv_writer.writerow(
                (
                    "world_timestamp",
                    "world_index",
                    "fixation_id",
                    "start_timestamp",
                    "duration",
                    "dispersion",
                    "norm_pos_x",
                    "norm_pos_y",
                    "x_scaled",
                    "y_scaled",
                    "on_surf",
                )
            )
            for world_idx, fixs_for_frame in enumerate(fixations_on_surf):
                world_idx += self.export_range[0]
                if fixs_for_frame:
                    for fix in fixs_for_frame:
                        csv_writer.writerow(
                            (
                                self.world_timestamps[world_idx],
                                world_idx,
                                fix["id"],
                                fix["timestamp"],
                                fix["duration"],
                                fix["dispersion"],
                                fix["norm_pos"][0],
                                fix["norm_pos"][1],
                                fix["norm_pos"][0] * surface.real_world_size["x"],
                                fix["norm_pos"][1] * surface.real_world_size["y"],
                                fix["on_surf"],
                            )
                        )
