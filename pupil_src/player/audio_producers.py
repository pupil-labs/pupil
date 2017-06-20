'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import av
import numpy as np
from plugin import Producer_Plugin_Base
from OpenGL.GL import *
from pyglui.cygl.utils import *

import logging
logger = logging.getLogger(__name__)


def dtype_from_format(fmt):
    fmt = fmt.replace('u', 'uint')
    fmt = fmt.replace('s', 'int')
    fmt = fmt.replace('flt', 'float32')
    fmt = fmt.replace('dbl', 'float64')
    fmt = fmt[:-1] if fmt[-1] == 'p' else fmt
    return np.dtype(fmt)


class Audio_From_Recording(Producer_Plugin_Base):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.wave_points = []
        audio_file_loc = os.path.join(g_pool.rec_dir, 'audio.mp4')
        audio_ts_loc = os.path.join(g_pool.rec_dir, 'audio_timestamps.npy')
        audio_format = None
        if os.path.exists(audio_file_loc) and os.path.exists(audio_ts_loc):
            audio_file = av.open(audio_file_loc)
            audio_ts = np.load(audio_ts_loc)
            ts_idx = 0
            for audio_packet in audio_file.demux():

                for frm in audio_packet.decode():
                    audio_plane = frm.planes[0]
                    if audio_format is None:
                        audio_format = dtype_from_format(frm.format.name)
                    audio = np.frombuffer(audio_plane.to_bytes(), dtype=audio_format)
                    if ts_idx >= len(audio_ts):
                        break
                    self.wave_points.append((audio_ts[ts_idx], audio.mean()*5))
                    ts_idx += 1

            self.notify_all({'subject': 'audio_changed'})
        else:
            logger.error('Could not find audio in recording.')
            self._alive = False

        if self.wave_points:
            self.wave_points = np.asarray(self.wave_points)
            std = self.wave_points[:, 1].std() * 25
            # in-place clipping to [- std, std]
            self.wave_points[:, 1].clip(min=-std, max=std, out=self.wave_points[:, 1])
            self.wave_points /= np.abs(self.wave_points[:, 1]).max()  # scale such that  all values lie between [-1, 1]
            # self.wave_points = self.wave_points.tolist()

        self.requires_display = True
        self.win_size = g_pool.capture.frame_size

    def gl_display(self):
        # `if self.wave_points:` raises ValueError if self.wave_points is a np.ndarray
        if len(self.wave_points) > 0:

            padding = 30.
            min_ts = self.wave_points[0, 0]
            max_ts = self.wave_points[-1, 0]
            # min_ts = self.g_pool.timestamps[0]
            # max_ts = self.g_pool.timestamps[-1]
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            width, height = self.win_size
            h_pad = padding * (max_ts - min_ts) / float(width)
            v_pad = padding * 2. / (height)
            # ranging from min_ts tp max_ts (horizontal) and -1 to 1 (vertical)
            glOrtho(-h_pad + min_ts, h_pad + max_ts, -v_pad, 1 + v_pad, -1, 1)


            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            glTranslatef(0, .03, 0)
            draw_polyline(self.wave_points, color=RGBA(0., 1., 0., .5))

            # for s in self.sections:
            #     cal_slc = parse_range(s['calibration_range'], max_ts)
            #     map_slc = parse_range(s['mapping_range'], max_ts)
            #     color = RGBA(*s['color'], .8)

            #     draw_polyline([(cal_slc.start, 0), (cal_slc.stop, 0)], color=color, line_type=GL_LINES, thickness=4)
            #     draw_polyline([(map_slc.start, 0), (map_slc.stop, 0)], color=color, line_type=GL_LINES, thickness=2)
            #     glTranslatef(0, .015, 0)

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

    def on_window_resize(self, window, w, h):
        self.win_size = w, h

    def get_init_dict(self):
        return {}
