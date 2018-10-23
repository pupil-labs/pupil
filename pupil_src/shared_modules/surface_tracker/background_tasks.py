import background_helper

# TODO clean this up!


def background_video_processor(video_file_path, callable, visited_list, seek_idx=-1):
    return background_helper.Task_Proxy(
        "Background Video Processor",
        video_processing_generator,
        (video_file_path, callable, seek_idx, visited_list),
    )


def video_processing_generator(video_file_path, callable, seek_idx, visited_list):
    import os
    import logging

    logger = logging.getLogger(__name__ + " with pid: " + str(os.getpid()))
    logger.debug("Started cacher process for Marker Detector")
    import video_capture

    class Global_Container(object):
        pass

    cap = video_capture.init_playback_source(
        Global_Container(), source_path=video_file_path, timing=None
    )

    visited_list = [
        True if not isinstance(x, bool) or x else False for x in visited_list
    ]

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
            logger.debug("Seeking to Frame {}".format(frame_idx))
            try:
                cap.seek_to_frame(frame_idx)
            except video_capture.FileSeekError:
                logger.warning("Could not evaluate frame: {}.".format(frame_idx))
                visited_list[frame_idx] = True  # this frame is now visited.
                return None

        try:
            frame = cap.get_frame()
        except video_capture.EndofVideoError:
            logger.warning("Could not evaluate frame: {}.".format(frame_idx))
            visited_list[frame_idx] = True
            return None
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
                "User required seek. Marker caching at Frame: {}".format(last_frame_idx)
            )

        next_frame_idx = next_unvisited_idx(last_frame_idx)

        if next_frame_idx is None:
            break
        else:
            res = handle_frame(next_frame_idx)
            visited_list[next_frame_idx] = True
            yield next_frame_idx, res


def background_data_processor(data, callable, visited_list, seek_idx=-1):
    return background_helper.Task_Proxy(
        "Background Data Processor",
        data_processing_generator,
        (data, callable, seek_idx, visited_list),
    )


def data_processing_generator(data, callable, seek_idx, visited_list):
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

        if not visited:
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
    surfaces, section, all_gaze_timestamps, all_gaze_events, camera_model
):
    return background_helper.Task_Proxy(
        "Background Data Processor",
        gaze_on_surface_generator,
        (surfaces, section, all_gaze_timestamps, all_gaze_events, camera_model),
    )
