import abc
import typing as T

from plugin import Plugin


class PupilDetectorPlugin(Plugin):

    ########## PupilDetectorPlugin API

    label = "Unnamed"
    identifier = "unnamed"

    @property
    @abc.abstractmethod
    def pupil_detector(self) -> PupilDetector:
        pass

    ##### Plugin API

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self._notification_handler = {
            "pupil_detector.broadcast_properties": self.handle_broadcast_properties_notification,
            "pupil_detector.set_property": self.handle_set_property_notification,
        }

    def recent_events(self, event):
        frame = event.get("frame")
        if not frame:
            return

        # TODO: Extract event handling logic from eye.py

        # Pupil ellipse detection
        event["pupil_detection_result"] = self.detect(
            frame=frame,
            user_roi=self.g_pool.u_r,
            visualize=self.g_pool.display_mode == "algorithm",
        )

    def on_notify(self, notification):
        subject = notification["subject"]
        for subject_prefix, handler in self._notification_handler.items():
            if subject.startswith(subject_prefix):
                handler(notification)

    def handle_broadcast_properties_notification(self, notification):
        target_process = notification.get("target", g_pool.process)
        eye_id = target_process[-1]  # either "0" or "1"
        should_respond = target_process == g_pool.process
        if should_respond:
            props = self.namespaced_detector_properties()
            properties_broadcast = {
                "subject": f"pupil_detector.properties.{eye_id}",
                **props,  # add properties to broadcast
            }
            self.notify_all(properties_broadcast)

    def handle_set_property_notification(self, notification):
        target_process = notification.get("target", g_pool.process)
        should_apply = target_process == g_pool.process

        if should_apply:
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
                        namespace, {property_name: property_value}
                    )
                elif property_name == "roi":
                    try:
                        # Modify the ROI with the values sent over network
                        minX, maxX, minY, maxY = property_value
                        self.g_pool.u_r.set(
                            [
                                max(g_pool.u_r.min_x, int(minX)),
                                max(g_pool.u_r.min_y, int(minY)),
                                min(g_pool.u_r.max_x, int(maxX)),
                                min(g_pool.u_r.max_y, int(maxY)),
                            ]
                        )
                    except ValueError as err:
                        raise ValueError(
                            "ROI needs to be list of 4 integers:"
                            "(minX, maxX, minY, maxY)"
                        ) from err
                else:
                    raise KeyError(
                        "Notification subject does not "
                        "specifiy detector type nor modify ROI."
                    )
                logger.debug(
                    "`{}` property set to {}".format(property_name, property_value)
                )
            except KeyError:
                logger.error("Malformed notification received")
                logger.debug(traceback.format_exc())
            except (ValueError, TypeError):
                logger.error("Invalid property or value")
                logger.debug(traceback.format_exc())

    ########## PupilDetector API

    ##### Legacy API

    def set_2d_detector_property(self, name: str, value: T.Any):
        return self.pupil_detector.set_2d_detector_property(name=name, value=value)

    def set_3d_detector_property(self, name: str, value: T.Any):
        return self.pupil_detector.set_3d_detector_property(name=name, value=value)

    ##### Core API

    def detect(self, frame, user_roi, visualize, pause_video: bool = False, **kwargs):
        return self.pupil_detector.detect(
            frame=frame,
            user_roi=user_roi,
            visualize=visualize,
            pause_video=pause_video,
            **kwargs,
        )

    def namespaced_detector_properties(self) -> dict:
        return self.pupil_detector.namespaced_detector_properties()

    def on_resolution_change(self, old_size, new_size):
        return self.pupil_detector.on_resolution_change(
            old_size=old_size, new_size=new_size
        )
