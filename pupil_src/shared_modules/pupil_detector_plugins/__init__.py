import logging
import typing as T

from .detector_2d_plugin import Detector2DPlugin
from .detector_3d_plugin import Detector3DPlugin
from .detector_base_plugin import PupilDetectorPlugin
from .detector_dummy_plugin import DetectorDummyPlugin

logger = logging.getLogger(__name__)


def available_detector_plugins() -> T.Tuple[
    PupilDetectorPlugin, T.List[PupilDetectorPlugin]
]:
    """Load and list available plugins, including default
    
    Returns:
        T.Tuple[PupilDetectorPlugin, T.List[PupilDetectorPlugin]]
        --  Return tuple of default plugin, and list available plugins.
            Default is required to be in the list of available plugins.
    """

    detector_plugins = [DetectorDummyPlugin, Detector2DPlugin, Detector3DPlugin]

    try:
        from py3d import Detector3DRefractionPlugin

        detector_plugins.append(Detector3DRefractionPlugin)
    except ImportError:
        logging.info("Refraction corrected 3D pupil detector not available")

    return Detector3DPlugin, detector_plugins
