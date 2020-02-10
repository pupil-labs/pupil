from .base_plugin import CalibrationChoreographyPlugin
from .screen_marker_plugin import ScreenMarkerChoreographyPlugin
from .single_marker_plugin import SingleMarkerChoreographyPlugin
from .natural_feature_plugin import NaturalFeatureChoreographyPlugin

# from .hmd_plugin import HMDChoreographyPlugin


def available_calibration_choreography_plugins():
    default_plugin = ScreenMarkerChoreographyPlugin
    available_plugins = list(
        CalibrationChoreographyPlugin.registered_choreographies_by_label().values()
    )
    return default_plugin, available_plugins
