"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import logging
import re
import traceback
import typing as T

from plugin import Plugin
from pupil_detectors import DetectorBase

logger = logging.getLogger(__name__)

EVENT_KEY = "pupil_detection_results"

PUPIL_DETECTOR_NOTIFICATION_SUBJECT_PATTERN = r"^pupil_detector\.(?P<action>[a-z_]+)$"


class DetectorPropertyProxy:
    """Wrapper around detector properties for easy UI coupling."""

    def __init__(self, detector):
        self.__dict__["detector"] = detector

    def __getattr__(self, key):
        return self.detector.get_properties()[key]

    def __setattr__(self, key, value):
        self.detector.update_properties({key: value})


class PupilDetectorPlugin(Plugin):
    label = "Unnamed"  # Used in eye -> general settings as selector

    pupil_detection_identifier = "unnamed"
    pupil_detection_method = "undefined"

    @property
    @abc.abstractmethod
    def pupil_detector(self) -> DetectorBase:
        pass

    @property
    def pupil_detector_properties(self) -> DetectorPropertyProxy:
        attr_name = "__pupil_detector_properties"
        if not hasattr(self, attr_name):
            property_proxy = DetectorPropertyProxy(self.pupil_detector)
            setattr(self, attr_name, property_proxy)
        return getattr(self, attr_name)

    @abc.abstractmethod
    def detect(self, frame, **kwargs):
        pass

    def create_pupil_datum(self, norm_pos, diameter, confidence, timestamp) -> dict:
        """"""
        eye_id = self.g_pool.eye_id

        # TODO: Assert arguments are valid

        # Create basic pupil datum with required fields
        datum = {}
        datum["id"] = eye_id
        datum["topic"] = f"pupil.{eye_id}.{self.pupil_detection_identifier}"
        datum["method"] = f"{self.pupil_detection_method}"
        datum["norm_pos"] = norm_pos
        datum["diameter"] = diameter
        datum["confidence"] = confidence
        datum["timestamp"] = timestamp
        return datum

    ### Plugin API

    uniqueness = "by_class"
    order = 0.1

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, is_on: bool):
        self._enabled = is_on
        for elem in self.menu:
            elem.read_only = not self.enabled

    def __init__(self, g_pool):
        super().__init__(g_pool)
        g_pool.pupil_detector = self
        self._recent_detection_result = None

        self._enabled = True

    def init_ui(self):
        self.add_menu()

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, event):
        frame = event.get("frame", None)

        if not frame or not self.enabled:
            self._recent_detection_result = None
            return

        # Detect if resolution changed
        self.__update_frame_size_if_changed(new_frame_size=(frame.width, frame.height))

        # Get results of previous detectors
        # TODO: This is currently used by Pye3D to get the results of the 2D detection
        previous_detection_results = event.get(EVENT_KEY, [])

        detection_result = self.detect(
            frame=frame,
            # TODO: workaround to get 2D data into pye3D for now
            previous_detection_results=previous_detection_results,
        )

        # Append the new detection result to the previous results
        event[EVENT_KEY] = previous_detection_results + [detection_result]

        # Save the most recent detection result for visualization
        self._recent_detection_result = detection_result

    def on_notify(self, notification):
        subject_match = re.match(
            PUPIL_DETECTOR_NOTIFICATION_SUBJECT_PATTERN, notification["subject"]
        )

        if subject_match:
            # {
            #   "subject": "pupil_detector.<action>",
            #   "eye_id": <eye_id: int>, # optinal
            #   "detector_plugin_class_name": <detector_plugin_class_name: str>, # optional
            #   ...
            # }

            action = subject_match["action"]

            this_eye_id = self.g_pool.eye_id
            this_plugin_class_name = type(self).__name__

            if "eye_id" in notification and notification["eye_id"] != this_eye_id:
                # Eye id doesn't match current process eye id - ignoring notification
                # NOTE: Missing value behaves like a match
                return

            if (
                "detector_plugin_class_name" in notification
                and notification["detector_plugin_class_name"] != this_plugin_class_name
            ):
                # Detector plugin class name doesn't match current plugin - ignoring notification
                # NOTE: Missing value behaves like a match
                return

            if action == "set_enabled":
                self.__safely_update_enabled(notification)

            elif action == "set_roi":
                self.__safely_update_roi(notification)

            elif action == "set_properties":
                self.__safely_update_properties(notification)

            elif action == "broadcast_properties":
                self.notify_all(
                    {
                        "subject": f"pupil_detector.properties.{this_eye_id}.{this_plugin_class_name}",
                        "values": self.pupil_detector.get_properties(),
                    }
                )

    def on_resolution_change(self, old_size, new_size):
        pass

    ### Private helpers

    def __safely_update_enabled(self, notification: dict):
        try:
            value = notification["value"]
        except KeyError:
            logger.error(f"Malformed notification: missing 'value' key")
            logger.debug(traceback.format_exc())
            return

        if not isinstance(value, bool):
            logger.error("Enabled value must be a bool")
            logger.error(f"Invalid value: {value}")
            return

        self.enabled = value
        logger.debug(f"enabled set to {value}")

    def __safely_update_roi(self, notification):
        try:
            value = notification["value"]
        except KeyError:
            logger.error(f"Malformed notification: missing 'value' key")
            logger.debug(traceback.format_exc())
            return

        try:
            minX, minY, maxX, maxY = value
            minX, minY, maxX, maxY = int(minX), int(minY), int(maxX), int(maxY)
        except (ValueError, TypeError) as err:
            # NOTE:
            #   - TypeError gets thrown when 'value' is not a tuple
            #   - TypeError gets thrown when 'minX', 'minY', 'maxX', or 'maxY' are not 'int's
            #   - ValueError gets throws when length of the tuple 'value' is not 4
            logger.error("ROI needs to be 4 integers: (minX, minY, maxX, maxY)")
            logger.error(f"Invalid value: {value}")
            logger.debug(traceback.format_exc())
            return

        # Apply very strict error checking here, although roi deal with invalid
        # values, so the user gets immediate feedback and does not wonder why
        # something did not work as expected.
        width, height = self.g_pool.roi.frame_size
        if not ((0 <= minX < maxX < width) and (0 <= minY < maxY <= height)):
            logger.error(
                "Received ROI with invalid dimensions!"
                f" (minX={minX}, minY={minY}, maxX={maxX}, maxY={maxY})"
                f" for frame size ({width} x {height})"
            )
            logger.error(f"Invalid value: {value}")
            return

        self.g_pool.roi.bounds = (minX, minY, maxX, maxY)
        logger.debug(f"roi set to {value}")

    def __safely_update_properties(self, notification):
        try:
            plugin_class_name = notification["detector_plugin_class_name"]
        except KeyError:
            logger.error(
                f"Malformed notification: missing 'detector_plugin_class_name' key"
            )
            logger.debug(traceback.format_exc())
            return

        if not isinstance(plugin_class_name, str):
            logger.error("Detector plugin class name must be a string")
            logger.error(f"Invalid value: {plugin_class_name}")
            return

        if plugin_class_name != type(self).__name__:
            # Detector plugin class name doesn't match current plugin - ignoring notification
            return

        try:
            values = notification["values"]
        except KeyError:
            logger.error(f"Malformed notification: missing 'values' key")
            logger.debug(traceback.format_exc())
            return

        if not isinstance(values, dict):
            logger.error(f"Invalid value: {values}")
            logger.debug(traceback.format_exc())
            return

        try:
            self.pupil_detector.update_properties(values)
        except (ValueError, TypeError):
            logger.error(f"Invalid value: {values}")
            logger.debug(traceback.format_exc())

    def __update_frame_size_if_changed(self, new_frame_size):
        attr_name = "__last_frame_size"

        old_frame_size = getattr(self, attr_name, None)

        if new_frame_size != old_frame_size:
            if old_frame_size is not None:
                self.on_resolution_change(old_frame_size, new_frame_size)
            setattr(self, attr_name, new_frame_size)
