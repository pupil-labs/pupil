'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import numpy as np


class CheckFrameStripes(object):
    def __init__(self, frame_size, num_check_frames=200):
        self.row_index = np.linspace(0, frame_size[0] - 1, 10, dtype=np.int)
        self.num_check_frames = num_check_frames
        self.num_local_optimum = []

    def __call__(self, frame):
        restart_flag = 0
        self.num_local_optimum.append(self.check_row(frame))
        if len(self.num_local_optimum) >= self.num_check_frames:
            stripes_ratio = np.mean(self.num_local_optimum) / len(self.row_index)
            restart_flag = 1 if stripes_ratio > 0.35 else -1
            self.num_local_optimum = []

        return restart_flag

    def check_row(self, frame):
        num_local_max = 0
        num_local_min = 0

        for i in self.row_index:
            arr = np.array(frame[i, :], dtype=np.int)

            local_max = \
            np.where(np.r_[False, True, arr[2:] > arr[:-2] + 20] & np.r_[arr[:-2] > arr[2:] + 20, True, False] == True)[
                0]
            num_local_max += len(local_max)
            local_min = \
            np.where(np.r_[False, True, arr[2:] + 20 < arr[:-2]] & np.r_[arr[:-2] + 20 < arr[2:], True, False] == True)[
                0]
            num_local_min += len(local_min)

        return num_local_max + num_local_min
