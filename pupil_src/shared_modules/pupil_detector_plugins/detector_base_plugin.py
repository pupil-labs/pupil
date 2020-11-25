"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc
import logging
import traceback
import typing as T

from pupil_detectors import DetectorBase

from plugin import Plugin

logger = logging.getLogger(__name__)

EVENT_KEY = "pupil_detection_results"


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
        self._update_frame_size_if_changed(new_frame_size=(frame.width, frame.height))

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
        subject_components = notification["subject"].split(".")

        if subject_components[0] == "pupil_detector":
            # {
            #   "suject": "pupil_detector.<action>",
            #   "target": <target_process>, # optinal
            #   ...
            # }

            action = subject_components[1]

            # Check if target process matches the current processes (missing value behaves like a match)
            does_process_match = ("target" not in notification) or (
                notification["target"] == self.g_pool.process
            )

            # TODO: Get identifier/class_name of the pupil detection plugin, and check if it matches current plugin

            if not does_process_match:
                return

            if action == "set_enabled":
                value = notification.get("value", None)
                self.__safely_update_enabled(value)

            # TODO: Change notification to "pupil_detector.set_roi"
            elif action == "set_property" and notification.get("name", None) == "roi":
                value = notification.get("value", None)
                self.__safely_update_roi(value)

            elif action == "set_property":
                name = notification.get("name", None)
                value = notification.get("value", None)
                self.__safely_update_property(name, value)

            elif action == "broadcast_properties":
                self.notify_all(
                    {
                        # TODO: Include pupil detection plugin identifier in subject or payload
                        "subject": f"pupil_detector.{self.g_pool.eye_id}.properties",
                        "value": self.pupil_detector.get_properties(),
                    }
                )

    def on_resolution_change(self, old_size, new_size):
        pass

    ### Private helpers

    def __safely_update_enabled(self, value):
        try:
            pass  # FIXME
        except (ValueError, TypeError):
            logger.error(f"Invalid enabled value ({value})")
            logger.debug(traceback.format_exc())
        else:
            self.enabled = value
            logger.debug(f"roi set to {value}")

    def __safely_update_roi(self, value):
        try:
            try:
                minX, minY, maxX, maxY = value
            except (ValueError, TypeError) as err:
                # NOTE: ValueError gets throws when length of the tuple does not
                # match. TypeError gets thrown when it is not a tuple.
                raise ValueError(
                    "ROI needs to be 4 integers: (minX, minY, maxX, maxY)"
                ) from err

            # Apply very strict error checking here, although roi deal with invalid
            # values, so the user gets immediate feedback and does not wonder why
            # something did not work as expected.
            width, height = self.g_pool.roi.frame_size
            if not ((0 <= minX < maxX < width) and (0 <= minY < maxY <= height)):
                raise ValueError(
                    "Received ROI with invalid dimensions!"
                    f" (minX={minX}, minY={minY}, maxX={maxX}, maxY={maxY})"
                    f" for frame size ({width} x {height})"
                )
        except (ValueError, TypeError):
            logger.error(f"Invalid roi value ({value})")
            logger.debug(traceback.format_exc())
        else:
            self.g_pool.roi.bounds = (minX, minY, maxX, maxY)
            logger.debug(f"roi set to {value}")

    def __safely_update_property(self, name, value):
        try:
            self.pupil_detector.update_properties({name: value})
        except (ValueError, TypeError):
            logger.error(f"Invalid property name ('{name}') or value ({value})")
            logger.debug(traceback.format_exc())
        else:
            logger.debug(f"'{name}' property set to {value}")

    def __update_frame_size_if_changed(self, new_frame_size):
        attr_name = "__last_frame_size"

        old_frame_size = getattr(self, attr_name, default=None)

        if new_frame_size != old_frame_size:
            if old_frame_size is not None:
                self.on_resolution_change(old_frame_size, new_frame_size)
            setattr(self, attr_name, new_frame_size)