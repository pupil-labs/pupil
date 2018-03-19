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


class Check_Frame_Stripes(object):
    def __init__(self, check_freq_init=0.2, check_freq_upperbound=5, check_freq_lowerbound=0.0001, factor=1.5):
        self.check_freq_init = check_freq_init
        self.check_freq = self.check_freq_init
        self.check_freq_upperbound = check_freq_upperbound
        self.check_freq_lowerbound = check_freq_lowerbound
        self.factor = factor

        self.last_check_timestamp = None
        self.len_row_index = 10
        self.len_col_index = 10

    def require_restart(self, frame):
        if self.last_check_timestamp is None:
            self.last_check_timestamp = frame.timestamp

        if frame.timestamp - self.last_check_timestamp > self.check_freq:
            self.last_check_timestamp = frame.timestamp
            if self.check_slice(frame.gray):
                self.check_freq = min(self.check_freq_init, self.check_freq)
                self.check_freq /= self.factor
            else:
                if self.check_freq < self.check_freq_upperbound:
                    self.check_freq = min(self.check_freq*self.factor, self.check_freq_upperbound)
            if self.check_freq < self.check_freq_lowerbound:
                return True

        return False

    def check_slice(self, frame_gray):
        num_local_max = np.zeros(2, dtype=np.int)
        num_local_min = np.zeros(2, dtype=np.int)

        row_index = np.random.randint(0, frame_gray.shape[0], size=self.len_row_index)
        col_index = np.random.randint(0, frame_gray.shape[1], size=self.len_col_index)
        for n in range(2):
            if n == 0:
                arrs = np.array(frame_gray[row_index, :], dtype=np.int)
            else:
                arrs = np.array(frame_gray[:, col_index], dtype=np.int)
                arrs = np.transpose(arrs)
            for arr in arrs:
                local_max = np.where(np.r_[False, True, arr[2:] > arr[:-2] + 30] & np.r_[arr[:-2] > arr[2:] + 30, True, False] == True)[0]
                num_local_max[n] += len(local_max)
                local_min = np.where(np.r_[False, True, arr[2:] + 30 < arr[:-2]] & np.r_[arr[:-2] + 30 < arr[2:], True, False] == True)[0]
                num_local_min[n] += len(local_min)

        num_local_optimum = num_local_max + num_local_min
        stripes_ratio = num_local_optimum[0] / self.len_row_index, num_local_optimum[1] / self.len_col_index

        return max(stripes_ratio) > 0.4
