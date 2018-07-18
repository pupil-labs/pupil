'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
# Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/functions/prior_box.py


import torch
import math
import itertools


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.img_size = cfg['input_size']
        self.num_priors = len(cfg['aspect_ratios'])
        self.aspect_ratios = cfg['aspect_ratios']
        self.feature_maps = cfg['feature_maps_reso']
        self.steps = [int(self.img_size / f + 1) for f in cfg['feature_maps_reso']]
        self.min_sizes = [self.img_size * f for f in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]]
        self.max_sizes = [self.img_size * f for f in [0.3, 0.45, 0.6, 0.75, 0.9, 1.05]]
        self.clip = True

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.img_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1; rel size: min_size
                s_k = self.min_sizes[k]/self.img_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1; rel size: sqrt(s_k * s_(k+1))
                s_k_prime = math.sqrt(s_k * (self.max_sizes[k] / self.img_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*math.sqrt(ar), s_k/math.sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
