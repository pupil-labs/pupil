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
    def __init__(self, check_freq_init=0.1, check_freq_upperbound=5, check_freq_lowerbound=0.00001, factor=0.8):
        self.check_freq_init = check_freq_init
        self.check_freq = self.check_freq_init
        self.check_freq_upperbound = check_freq_upperbound
        self.check_freq_lowerbound = check_freq_lowerbound
        self.factor = factor
        self.factor_mul = 1.1

        self.last_check_timestamp = None
        self.len_row_index = 8
        self.len_col_index = 8
        self.row_index = None
        self.col_index = None

    def require_restart(self, frame):
        if self.last_check_timestamp is None:
            self.last_check_timestamp = frame.timestamp

        if frame.timestamp - self.last_check_timestamp > self.check_freq:
            self.last_check_timestamp = frame.timestamp
            res = self.check_slice(frame.gray)

            if res is True:
                self.check_freq = min(self.check_freq_init, self.check_freq) * self.factor
                if self.check_freq < self.check_freq_lowerbound:
                    return True
            elif res is False:
                if self.check_freq < self.check_freq_upperbound:
                    self.check_freq = min(self.check_freq*self.factor_mul, self.check_freq_upperbound)

        return False

    def check_slice(self, frame_gray):
        num_local_optimum = [0, 0]
        if self.row_index is None:
            self.row_index = np.linspace(8, frame_gray.shape[0]-8, num=self.len_row_index, dtype=np.int)
            self.col_index = np.linspace(8, frame_gray.shape[1]-8, num=self.len_col_index, dtype=np.int)
        for n in [0, 1]:
            if n == 0:
                arrs = np.array(frame_gray[self.row_index, :], dtype=np.int)
            else:
                arrs = np.array(frame_gray[:, self.col_index], dtype=np.int)
                arrs = np.transpose(arrs)

            local_max_union = set()
            local_min_union = set()
            for arr in arrs:
                local_max = set(np.where(np.r_[False, True, arr[2:] > arr[:-2] + 30] & np.r_[arr[:-2] > arr[2:] + 30, True, False] == True)[0])
                local_min = set(np.where(np.r_[False, True, arr[2:] + 30 < arr[:-2]] & np.r_[arr[:-2] + 30 < arr[2:], True, False] == True)[0])
                num_local_optimum[n] += len(local_max_union.intersection(local_max)) + len(local_min_union.intersection(local_min))
                if sum(num_local_optimum) >= 3:
                    return True
                local_max_union = local_max_union.union(local_max)
                local_min_union = local_min_union.union(local_min)

        if sum(num_local_optimum) == 0:
            return False
        else:
            return None
