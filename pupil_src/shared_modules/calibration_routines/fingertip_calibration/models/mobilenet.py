"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


import torch.nn as nn


def conv(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def depth_sep_conv(inp, oup, stride):
    return nn.Sequential(
        # dw
        nn.Conv2d(
            inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False
        ),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        # pw
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def mobilenet(width_multiplier=1.0):
    """MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    See: https://arxiv.org/abs/1704.04861
    for more details.

    Args:
        width_multiplier: the parameter to thin a network uniformly at each layer

    """
    layers = []
    layers += [conv(3, int(32 * width_multiplier), 2)]
    layers += [
        depth_sep_conv(int(32 * width_multiplier), int(64 * width_multiplier), 1)
    ]
    layers += [
        depth_sep_conv(int(64 * width_multiplier), int(128 * width_multiplier), 2)
    ]
    layers += [
        depth_sep_conv(int(128 * width_multiplier), int(128 * width_multiplier), 1)
    ]
    layers += [
        depth_sep_conv(int(128 * width_multiplier), int(256 * width_multiplier), 2)
    ]
    layers += [
        depth_sep_conv(int(256 * width_multiplier), int(256 * width_multiplier), 1)
    ]
    layers += [
        depth_sep_conv(int(256 * width_multiplier), int(512 * width_multiplier), 2)
    ]
    layers += [
        depth_sep_conv(int(512 * width_multiplier), int(512 * width_multiplier), 1)
    ]
    layers += [
        depth_sep_conv(int(512 * width_multiplier), int(512 * width_multiplier), 1)
    ]
    layers += [
        depth_sep_conv(int(512 * width_multiplier), int(512 * width_multiplier), 1)
    ]
    layers += [
        depth_sep_conv(int(512 * width_multiplier), int(512 * width_multiplier), 1)
    ]
    layers += [
        depth_sep_conv(int(512 * width_multiplier), int(512 * width_multiplier), 1)
    ]
    layers += [
        depth_sep_conv(int(512 * width_multiplier), int(1024 * width_multiplier), 2)
    ]
    layers += [
        depth_sep_conv(int(1024 * width_multiplier), int(1024 * width_multiplier), 1)
    ]

    return layers
