import logging

import player_methods as pm

from video_capture import File_Source, EndofVideoError

logger = logging.getLogger(__name__)


class _Empty:
    """Replacement for actual g_pool object"""


class FrameFetcher:
    __slots__ = ("source", "current_frame")

    def __init__(self, video_path):
        try:
            self.source = File_Source(_Empty(), source_path=video_path, timing=None)
        except AssertionError as err:
            raise FileNotFoundError(video_path) from err
        self.current_frame = self.source.get_frame()

    def initialised(self):
        return self.source.initialised

    def closest_frame_to_ts(self, ts):
        closest_idx = pm.find_closest(self.source.timestamps, ts)
        return self.frame_for_idx(closest_idx)

    def frame_for_idx(self, requested_frame_idx):
        if requested_frame_idx != self.current_frame.index:
            if requested_frame_idx == self.source.get_frame_index() + 2:
                # if we just need to seek by one frame,
                # its faster to just read one and and throw it away.
                self.source.get_frame()
            if requested_frame_idx != self.source.get_frame_index() + 1:
                self.source.seek_to_frame(int(requested_frame_idx))

            try:
                self.current_frame = self.source.get_frame()
            except EndofVideoError:
                logger.info("End of video {}.".format(self.source.source_path))
        return self.current_frame
