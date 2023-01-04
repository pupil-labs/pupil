"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from .base_plugin import (
    CalibrationChoreographyPlugin,
    ChoreographyAction,
    ChoreographyMode,
    ChoreographyNotification,
)


def import_builtin_choreography_plugins():
    from .hmd_plugin import HMD3DChoreographyPlugin
    from .natural_feature_plugin import NaturalFeatureChoreographyPlugin
    from .screen_marker_plugin import ScreenMarkerChoreographyPlugin
    from .single_marker_plugin import SingleMarkerChoreographyPlugin


def available_calibration_choreography_plugins():
    import_builtin_choreography_plugins()
    return list(
        CalibrationChoreographyPlugin.registered_choreographies_by_label().values()
    )


def default_calibration_choreography_plugin(app: str):
    registered = CalibrationChoreographyPlugin.registered_choreographies_by_label()
    if app == "capture":
        return registered["Screen Marker Calibration"]  # ScreenMarkerChoreographyPlugin
    if app == "service":
        return registered["HMD 3D Calibration"]  # HMD3DChoreographyPlugin
    if app == "player":
        raise NotImplementedError()
    raise ValueError(f'Unknown app "{app}"')


def patch_loaded_plugins_with_choreography_plugin(loaded_plugins, app: str):

    default_choreo_class = default_calibration_choreography_plugin(app=app)
    default_choreo_name = default_choreo_class.__name__

    available_choreo_classes = available_calibration_choreography_plugins()
    available_choreo_classes_by_name = {p.__name__: p for p in available_choreo_classes}

    loaded_plugins_contains_choreo_plugin = False

    for i, (plugin_name, _) in enumerate(list(loaded_plugins)):
        choreo_class = available_choreo_classes_by_name.get(plugin_name, None)

        if choreo_class is None:
            continue

        loaded_plugins_contains_choreo_plugin = True

        if not choreo_class.is_session_persistent:
            # If the choreography class is not session persistent,
            # replace it with the default choreography class
            loaded_plugins[i] = (default_choreo_name, {})

    if not loaded_plugins_contains_choreo_plugin:
        loaded_plugins.append((default_choreo_name, {}))

    return loaded_plugins
