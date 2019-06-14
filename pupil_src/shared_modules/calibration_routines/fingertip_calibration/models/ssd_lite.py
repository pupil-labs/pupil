"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
# Adapted from https://github.com/ShuangXieIrene/ssds.pytorch/blob/master/lib/modeling/ssds/ssd_lite.py &
# https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py


import torch
import torch.nn as nn

from calibration_routines.fingertip_calibration.models.layers.functions import (
    prior_box,
    detection,
)
from calibration_routines.fingertip_calibration.models import mobilenet


class SSDLite(nn.Module):
    """SSD: Single Shot MultiBox Detector
    See: https://arxiv.org/pdf/1512.02325.pdf & 
    https://arxiv.org/pdf/1611.10012.pdf &
    https://arxiv.org/pdf/1801.04381.pdf
    for more details.

    Args:
        cfg: configurations of the network
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, cfg, base, extras, head):
        super(SSDLite, self).__init__()
        self.cfg = cfg
        self.mobilenet = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)
        self.detect = detection.Detect(cfg)

        with torch.no_grad():
            self.priors = prior_box.PriorBox(self.cfg).forward()

    def forward(self, x):
        """Applies network layers on input image(s) x.

        Args:
            x: batch of images. Shape: [batch, 3, img_size, img_size].

        Return:
            tensor of output predictions for confidence score and corresponding locations.
            Shape: [batch, num_classes, topk, 5]
        """
        sources, loc, conf = [list() for _ in range(3)]

        # apply bases layers and cache source layer outputs
        for k in range(len(self.mobilenet)):
            x = self.mobilenet[k](x)
            if k in self.cfg["feature_maps_layer"]:
                sources.append(x)

        # apply extra layers and cache source layer outputs
        for v in self.extras:
            x = v(x)
            sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        #  get output predictions for confidence score and locations.
        output = self.detect(
            loc.view(loc.size(0), -1, 4),  # loc preds
            self.softmax(
                conf.view(conf.size(0), -1, self.cfg["num_classes"])
            ),  # conf preds
            self.priors.type((x.detach()).type()),  # default boxes
        )
        return output


def add_extras(cfg):
    extra_layers = []
    in_channels = None
    for layer, out_channels in zip(
        cfg["feature_maps_layer"], cfg["feature_maps_channel"]
    ):
        if layer == "S":
            extra_layers += [
                mobilenet.depth_sep_conv(in_channels, out_channels, stride=2)
            ]
        in_channels = out_channels

    return extra_layers


def add_head(cfg):
    mbox = [len(ar) + 2 for ar in cfg["aspect_ratios"]]
    loc_layers = []
    conf_layers = []
    for in_channels, num_box in zip(cfg["feature_maps_channel"], mbox):
        loc_layers += [nn.Conv2d(in_channels, num_box * 4, kernel_size=3, padding=1)]
        conf_layers += [
            nn.Conv2d(
                in_channels, num_box * cfg["num_classes"], kernel_size=3, padding=1
            )
        ]

    return loc_layers, conf_layers


def build_ssd_lite(cfg):
    width_multiplier = 0.25
    cfg["num_classes"] = 2
    cfg["aspect_ratios"] = [
        [1 / 2, 1 / 3],
        [1 / 2, 1 / 3],
        [1 / 2, 1 / 3],
        [1 / 2, 1 / 3],
        [1 / 2, 1 / 3],
        [1 / 2, 1 / 3],
    ]
    cfg["feature_maps_reso"] = [
        int(cfg["input_size"] / d + 1) for d in [16, 32, 64, 128, 256, 512]
    ]
    cfg["feature_maps_channel"] = [
        int(d * width_multiplier) for d in [512, 1024, 512, 256, 256, 128]
    ]
    cfg["feature_maps_layer"] = [11, 13, "S", "S", "S", "S"]

    return SSDLite(
        cfg=cfg,
        base=mobilenet.mobilenet(width_multiplier),
        extras=add_extras(cfg),
        head=add_head(cfg),
    )
