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


class PropertyProxy:
    """Wrapper around detector properties for easy UI coupling."""

    def __init__(self, detector):
        self.__dict__["detector"] = detector

    def __getattr__(self, namespaced_key):
        namespace, key = namespaced_key.split(".")
        return self.detector.get_properties()[namespace][key]

    def __setattr__(self, namespaced_key, value):
        namespace, key = namespaced_key.split(".")
        self.detector.update_properties({namespace: {key: value}})


class PupilDetectorPlugin(Plugin):
    label = "Unnamed"  # Used in eye -> general settings as selector
    # Used to select correct detector on set_pupil_detection_enabled:
    identifier = "unnamed"
    order = 0.1

    @property
    @abc.abstractmethod
    def pupil_detector(self) -> DetectorBase:
        pass

    def __init__(self, g_pool):
        super().__init__(g_pool)
        g_pool.pupil_detector = self
        self._recent_detection_result = None
        self._notification_handler = {
            "pupil_detector.broadcast_properties": self.handle_broadcast_properties_notification,
            "pupil_detector.set_property": self.handle_set_property_notification,
            "set_pupil_detection_enabled": self.handle_set_pupil_detection_enabled_notification,
        }
        self._last_frame_size = None
        self._enabled = True

    def init_ui(self):
        self.add_menu()

    def deinit_ui(self):
        self.remove_menu()

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value
        for elem in self.menu:
            elem.read_only = not self.enabled

    def recent_events(self, event):
        if not self.enabled:
            self._recent_detection_result = None
            return

        frame = event.get("frame")
        if not frame:
            self._recent_detection_result = None
            return

        frame_size = (frame.width, frame.height)
        if frame_size != self._last_frame_size:
            if self._last_frame_size is not None:
                self.on_resolution_change(self._last_frame_size, frame_size)
            self._last_frame_size = frame_size

        # TODO: The following sections with internal_2d_raw_data are for allowing dual
        # detectors until pye3d is finished. Then we should cleanup this section!

        # this is only revelant when running the 3D detector, for 2D we just ignore the
        # additional parameter
        detection_result = self.detect(
            frame=frame, internal_raw_2d_data=event.get("internal_2d_raw_data", None)
        )

        # if we are running the 2D detector, we might get internal data that we don't
        # want published, so we remove it from the dict
        if "internal_2d_raw_data" in detection_result:
            event["internal_2d_raw_data"] = detection_result.pop("internal_2d_raw_data")

        if EVENT_KEY in event:
            event[EVENT_KEY].append(detection_result)
        else:
            event[EVENT_KEY] = [detection_result]

        self._recent_detection_result = detection_result

    @abc.abstractmethod
    def detect(self, frame):
        pass

    def on_notify(self, notification):
        subject = notification["subject"]
        for subject_prefix, handler in self._notification_handler.items():
            if subject.startswith(subject_prefix):
                handler(notification)

    def handle_broadcast_properties_notification(self, notification):
        target_process = notification.get("target", self.g_pool.process)
        should_respond = target_process == self.g_pool.process
        if should_respond:
            props = self.namespaced_detector_properties()
            properties_broadcast = {
                "subject": f"pupil_detector.properties.{self.g_pool.eye_id}",
                **props,  # add properties to broadcast
            }
            self.notify_all(properties_broadcast)

    def handle_set_property_notification(self, notification):
        target_process = notification.get("target", self.g_pool.process)
        if target_process != self.g_pool.process:
            return

        try:
            property_name = notification["name"]
            property_value = notification["value"]
            subject_components = notification["subject"].split(".")
            # len(pupil_detector.properties) is at least 2 due to the subject prefix
            # being pupil_detector.set_property. The third component is the optional
            # namespace of the property.
            if len(subject_components) > 2:
                namespace = subject_components[2]
                self.pupil_detector.update_properties(
                    {namespace: {property_name: property_value}}
                )
            elif property_name == "roi":
                # Modify the ROI with the values sent over network

                try:
                    minX, minY, maxX, maxY = property_value
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

                self.g_pool.roi.bounds = (minX, minY, maxX, maxY)

            else:
                raise KeyError(
                    "Notification subject does not "
                    "specifiy detector type nor modify ROI."
                )
            logger.debug(f"'{property_name}' property set to {property_value}")
        except KeyError:
            logger.error("Malformed notification received")
            logger.debug(traceback.format_exc())
        except (ValueError, TypeError):
            logger.error("Invalid property or value")
            logger.debug(traceback.format_exc())

    def handle_set_pupil_detection_enabled_notification(self, notification):
        is_on = notification["value"]
        self.enabled = is_on

    def namespaced_detector_properties(self) -> dict:
        return self.pupil_detector.get_properties()

    def on_resolution_change(self, old_size, new_size):
        properties = self.pupil_detector.get_properties()
        properties["2d"]["pupil_size_max"] *= new_size[0] / old_size[0]
        properties["2d"]["pupil_size_min"] *= new_size[0] / old_size[0]
        self.pupil_detector.update_properties(properties)
