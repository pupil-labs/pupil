"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""


# stdlib
import os
import abc
import csv
import enum
import typing
import logging
import traceback
import operator
import functools
import itertools
import collections
import bisect
from typing import _Protocol as Protocol


# local
from tasklib import interface
from tasklib.background.task import BackgroundGeneratorFunction
from tasklib.background.patches import Patch, IPCLoggingPatch
from tasklib.manager import PluginTaskManager
import file_methods as fm
import player_methods as pm

import methods
from plugin import Analysis_Plugin_Base
from observable import Observable
import video_capture as vc

# third-party
import nslr_hmm
import numpy as np
import cv2
from pyglui import ui
from pyglui.cygl.utils import RGBA, draw_circle
from pyglui.pyfontstash import fontstash

