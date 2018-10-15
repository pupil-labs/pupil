"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections
import random

import numpy as np

from surface_tracker_future.surface import Surface


class Surface_Online(Surface):
    def __init__(self, init_dict=None):
        super().__init__(init_dict=init_dict)
        self.uid = random.randint(0, 1e6)

        self.name = "unknown"
        self.real_world_size = {"x": 1., "y": 1.}

        # We store the surface state in two versions: once computed with the
        # undistorted scene image and once with the still distorted scene image. The
        # undistorted state is used to map gaze onto the surface, the distorted one
        # is used for visualization. This is necessary because surface corners
        # outside of the image can not be re-distorted for visualization correctly.
        # Instead the slightly wrong but correct looking distorted version is
        # used for visualization.
        self.reg_markers_undist = {}
        self.reg_markers_dist = {}
        self.img_to_surf_trans = None
        self.surf_to_img_trans = None
        self.dist_img_to_surf_trans = None
        self.surf_to_dist_img_trans = None

        self.detected = False
        self.num_detected_markers = []

        self._required_obs_per_marker = 5
        self._avg_obs_per_marker = 0
        self.build_up_status = 0

        # TODO online specific
        self.gaze_history_length = 1
        self.gaze_history = collections.deque()
        self.heatmap = np.ones((1, 1), dtype=np.uint8)
        self.heatmap_detail = .2
        self.heatmap_min_data_confidence = 0.6

        if init_dict is not None:
            self.load_from_dict(init_dict)
