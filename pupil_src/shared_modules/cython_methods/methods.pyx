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
cimport numpy as np


def cumhist_color_map16(np.ndarray[np.uint16_t, ndim=2] depth_buffer):
    cdef int r, c, i, f, width, height
    cdef np.uint16_t d
    height = depth_buffer.shape[0]
    width = depth_buffer.shape[1]

    cdef np.ndarray[np.uint32_t, ndim=1] cumhist = np.zeros(0x10000, dtype=np.uint32)
    cdef np.ndarray[np.uint8_t, ndim=3] rgb_image = np.empty((height, width, 3), dtype=np.uint8)

    for r in range(height):
        for c in range(width):
            cumhist[depth_buffer[r, c]] += 1

    for i in range(2, 0x10000):
        cumhist[i] += cumhist[i-1]

    for r in range(height):
        for c in range(width):
            d = depth_buffer[r, c]
            if d != 0:
                # 0-255 based on histogram location
                f = cumhist[d] * 255 / cumhist[0xFFFF]
                rgb_image[r, c, 0] = f
                rgb_image[r, c, 1] = 0
                rgb_image[r, c, 2] = 255 - f
            else:
                rgb_image[r, c, 0] = 0
                rgb_image[r, c, 1] = 5
                rgb_image[r, c, 2] = 20

    return rgb_image
