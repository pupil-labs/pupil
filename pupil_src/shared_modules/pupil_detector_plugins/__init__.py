"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import typing as T

from .detector_2d_plugin import Detector2DPlugin
from .detector_3d_plugin import Detector3DPlugin
from .detector_base_plugin import PupilDetectorPlugin, EVENT_KEY

logger = logging.getLogger(__name__)


def available_detector_plugins() -> T.Tuple[
    PupilDetectorPlugin, PupilDetectorPlugin, T.List[PupilDetectorPlugin]
]:
    """Load and list available plugins, including default
    
    Returns tuple of default2D, default3D, and list of all detectors.
    """

    all_plugins = [Detector2DPlugin, Detector3DPlugin]
    default2D = Detector2DPlugin
    default3D = Detector3DPlugin

    try:
        from py3d import Detector3DRefractionPlugin

        all_plugins.append(Detector3DRefractionPlugin)
    except ImportError:
        logging.info("Refraction corrected 3D pupil detector not available")

    return default2D, default3D, all_plugins
