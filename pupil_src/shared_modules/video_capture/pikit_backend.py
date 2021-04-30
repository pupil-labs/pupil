import logging
import traceback

import av
import camera_models as cm
import gl_utils
import numpy as np
import pikit
from pyglui import cygl

from .base_backend import Base_Source, Playback_Source
from .file_backend import Frame

logger = logging.getLogger(__name__)


class Pikit_Source(Playback_Source, Base_Source):
    def __init__(
        self,
        g_pool,
        *args,
        **kwargs,
    ):
        super().__init__(g_pool, *args, **kwargs)
        self._rec = pikit.Recording(g_pool.rec_dir, use_world_time_files=True)
        self._timestamps_ns = np.concatenate(
            [part.times for part in self._rec.world.padded_sensor_part_readers]
        )
        self.timestamps = self.invisible_to_player_time(self._timestamps_ns)
        self.current_frame_idx = 0
        self.target_frame_idx = 0

        self._intrinsics = cm.Camera_Model.from_default("PI world v1", (1088, 1080))

    def invisible_to_player_time(self, ts):
        return (ts - self._rec.start_epoch_ns) / 1e9

    @property
    def name(self):
        return str(self.g_pool.rec_dir)

    @property
    def frame_size(self):
        """Summary
        Returns:
            tuple: 2-element tuple containing width, height
        """
        return 1080, 1088

    @property
    def frame_rate(self):
        """
        Returns:
            int/float: Frame rate
        """
        return 30.0

    @property
    def jpeg_support(self):
        """
        Returns:
            bool: Source supports jpeg data
        """
        return False

    def recent_events(self, events):
        """Returns None

        Adds events['frame']=Frame(args)
            Frame: Object containing image and time information of the current
            source frame.
        """
        try:
            last_index = self._recent_frame.index
        except AttributeError:
            # Get a frame at beginnning
            last_index = -1
        # Seek Frame
        frame = None
        pbt = self.g_pool.seek_control.current_playback_time
        ts_idx = self.g_pool.seek_control.ts_idx_from_playback_time(pbt)
        if ts_idx == last_index:
            frame = self._recent_frame.copy()
        elif ts_idx < last_index or ts_idx > last_index + 1:
            self.seek_to_frame(ts_idx)

        try:
            frame = frame or self.get_frame()
        except StopIteration:
            logger.info("No more video found")
            self.g_pool.seek_control.play = False
            frame = self._recent_frame.copy()
        self.g_pool.seek_control.end_of_seek()
        events["frame"] = frame
        self._recent_frame = frame

    def get_frame(self) -> Frame:
        pikit_frame = next(iter(self._rec.world))
        ts = self.invisible_to_player_time(pikit_frame.timestamp.epoch_ns)
        self.current_frame_idx = self.target_frame_idx
        self.target_frame_idx += 1
        return Frame(ts, pikit_frame.av_frame, self.current_frame_idx)

    def gl_display(self):
        if self._recent_frame is not None:
            frame = self._recent_frame
            if (
                frame.yuv_buffer is not None
                # TODO: Find a better solution than this:
                and getattr(self.g_pool, "display_mode", "") != "algorithm"
            ):
                self.g_pool.image_tex.update_from_yuv_buffer(
                    frame.yuv_buffer, frame.width, frame.height
                )
            else:
                self.g_pool.image_tex.update_from_ndarray(frame.bgr)
            gl_utils.glFlush()
        should_flip = getattr(self.g_pool, "flip", False)
        gl_utils.make_coord_system_norm_based(flip=should_flip)
        self.g_pool.image_tex.draw()
        if not self.online:
            cygl.utils.draw_gl_texture(np.zeros((1, 1, 3), dtype=np.uint8), alpha=0.4)
        gl_utils.make_coord_system_pixel_based(
            (self.frame_size[1], self.frame_size[0], 3), flip=should_flip
        )

    def seek_to_frame(self, frame_idx):
        self._rec.world.seek(self._timestamps_ns[frame_idx])

    def get_frame_index(self):
        return int(self.current_frame_idx)
