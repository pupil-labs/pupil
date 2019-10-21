import abc
import logging
import traceback
import typing as T

from pupil_detectors import DetectorBase

from plugin import Plugin

logger = logging.getLogger(__name__)


class PupilDetectorPlugin(Plugin):
    label = "Unnamed"  # Used in eye -> general settings as selector
    # Used to select correct detector on set_detection_mapping_mode:
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
        }

    def recent_events(self, event):
        frame = event.get("frame")
        if not frame:
            self._recent_detection_result = None
            return

        detection_result = self.detect(frame=frame)
        event["pupil_detection_result"] = detection_result
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
                    minX, maxX, minY, maxY = property_value
                except (ValueError, TypeError) as err:
                    # NOTE: ValueError gets throws when length of the tuple does not
                    # match. TypeError gets thrown when it is not a tuple.
                    raise ValueError(
                        "ROI needs to be 4 integers: (minX, maxX, minY, maxY)"
                    ) from err
                if minX > maxX or minY > maxY:
                    raise ValueError("ROI malformed: minX > maxX or minY > maxY!")
                ui_roi = self.g_pool.u_r
                ui_roi.lX = max(ui_roi.min_x, int(minX))
                ui_roi.lY = max(ui_roi.min_y, int(minY))
                ui_roi.uX = min(ui_roi.max_x, int(maxX))
                ui_roi.uY = min(ui_roi.max_y, int(maxY))
            else:
                raise KeyError(
                    "Notification subject does not "
                    "specifiy detector type nor modify ROI."
                )
            logger.debug(f"`{property_name}` property set to {property_value}")
        except KeyError:
            logger.error("Malformed notification received")
            logger.debug(traceback.format_exc())
        except (ValueError, TypeError):
            logger.error("Invalid property or value")
            logger.debug(traceback.format_exc())

    def set_2d_detector_property(self, name: str, value: T.Any):
        return self.pupil_detector.set_2d_detector_property(name=name, value=value)

    def set_3d_detector_property(self, name: str, value: T.Any):
        return self.pupil_detector.set_3d_detector_property(name=name, value=value)

    def namespaced_detector_properties(self) -> dict:
        return self.pupil_detector.namespaced_detector_properties()

    def on_resolution_change(self, old_size, new_size):
        return self.pupil_detector.on_resolution_change(
            old_size=old_size, new_size=new_size
        )
