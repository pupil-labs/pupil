"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import traceback
import typing as T

import pupil_detectors
from version_utils import parse_version

from .detector_2d_plugin import Detector2DPlugin
from .detector_base_plugin import EVENT_KEY, PupilDetectorPlugin

logger = logging.getLogger(__name__)

required_version_str = "2.0.0"
if parse_version(pupil_detectors.__version__) < parse_version(required_version_str):
    msg = (
        f"This version of Pupil requires pupil_detectors >= {required_version_str}."
        f" You are running with pupil_detectors == {pupil_detectors.__version__}."
        f" Please upgrade to a newer version!"
    )
    logger.error(msg)
    raise RuntimeError(msg)


def available_detector_plugins() -> T.List[T.Type[PupilDetectorPlugin]]:
    """Load and list available plugins

    Returns list of all detectors.
    """

    all_plugins: T.List[T.Type[PupilDetectorPlugin]] = [Detector2DPlugin]

    try:
        from .pye3d_plugin import Pye3DPlugin
    except ImportError:
        logger.warning("Refraction corrected 3D pupil detector not available!")
        logger.debug(traceback.format_exc())
    else:
        logger.debug("Using refraction corrected 3D pupil detector.")
        all_plugins.append(Pye3DPlugin)

    return all_plugins
