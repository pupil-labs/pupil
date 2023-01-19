import logging
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import uvc

from .definitions import (
    MODULE_SPEC,
    SCENE_CAM_SPEC,
    CameraSpec,
    GrayFrameProtocol,
    SplitSharedFrame,
    UVCDeviceInfo,
)


class NeonCameraInterface:
    def __init__(self, camera: CameraSpec, camera_reinit_timeout: float = 3) -> None:
        import uvc

        self.spec = camera

        self.logger = logging.getLogger(__name__ + ".camera")
        self._uvc_capture = None

        try:
            self._uvc_capture = self.init_connected_device(self.spec)
        except Exception as err:
            self.logger.warning(f"Camera init failed with {type(err).__name__}: {err}")

        self.timestamp_fn = uvc.get_time_monotonic
        self.last_frame_timestamp: float = self.timestamp_fn()

        self.camera_reinit_timeout = camera_reinit_timeout

    def __str__(self) -> str:
        return f"<{type(self).__name__} connected={self._uvc_capture is not None}>"

    @staticmethod
    def init_connected_device(
        camera: CameraSpec, uid: Optional[str] = None
    ) -> uvc.Capture:
        if uid is None:
            uid = NeonCameraInterface.find_connected_device_uid(camera)
            if uid is None:
                raise OSError(f"No matching camera with spec={camera} found")
        capture = uvc.Capture(uid)
        capture.bandwidth_factor = camera.bandwidth_factor
        for mode in capture.available_modes:
            if (mode.width, mode.height, mode.fps) == (
                camera.width,
                camera.height,
                camera.fps,
            ):
                capture.frame_mode = mode
                capture.get_frame(5.0)  # starts the stream
                return capture
        capture.close()
        raise OSError(f"None of the available modes matched: {capture.available_modes}")

    @staticmethod
    def any_device_connected() -> bool:
        return bool(NeonCameraInterface.find_all_connected_device_uids())

    @staticmethod
    def find_connected_device_uid(spec: CameraSpec) -> Optional[str]:
        try:
            return NeonCameraInterface.find_all_connected_device_uids((spec,))[0][1]
        except IndexError:
            return None

    @staticmethod
    def find_all_connected_device_uids(
        specs: Iterable[CameraSpec] = (SCENE_CAM_SPEC, MODULE_SPEC)
    ) -> List[Tuple[CameraSpec, str]]:
        devices: List[UVCDeviceInfo] = uvc.device_list()  # type: ignore
        found: List[Tuple[CameraSpec, str]] = []
        for device in devices:
            try:
                spec = next(spec for spec in specs if spec.matches_device(device))
                found.append((spec, device["uid"]))
            except StopIteration:
                pass
        return found

    def get_shared_frame(self, timeout: float = 0.0) -> Optional[GrayFrameProtocol]:
        """
        timeout values:
            -1.0 -> return immediately if no frame is available
             0.0 -> wait indefinitely until a frame becomes available
            >0.0 -> wait a maximum of `timeout` seconds until a frame becomes available
        """
        try:
            if self._uvc_capture is None:
                if timeout is not None:
                    time.sleep(timeout)
                raise OSError("No camera initialized")

            frame = self._uvc_capture.get_frame(timeout=timeout)
            frame.timestamp = self.last_frame_timestamp = self.timestamp_fn()
            return frame
        except (TimeoutError, OSError) as err:
            if (
                self.timestamp_fn() - self.last_frame_timestamp
                > self.camera_reinit_timeout
            ):
                self.logger.debug(
                    f"{self.camera_reinit_timeout} seconds since last frame or camera "
                    f"reinit ({type(err).__name__}({err})). Attempting reinit..."
                )
                self.close()
                try:
                    self._uvc_capture = self.init_connected_device(self.spec)
                except Exception as err:
                    self.logger.debug(
                        f"Camera reinit failed with {type(err).__name__}: {err}"
                    )
                self.last_frame_timestamp = self.timestamp_fn()

    @staticmethod
    def split_shared_frame(
        shared_frame: Optional["GrayFrameProtocol"],
    ) -> Optional[SplitSharedFrame]:
        if shared_frame is None:
            return
        gray_frame_cached = shared_frame.gray
        if not shared_frame.data_fully_received:
            logging.getLogger(__name__ + ".camera").debug(
                "Frame data not fully received. Dropping."
            )
            return

        width_half = gray_frame_cached.shape[1] // 2
        left_data = gray_frame_cached[:, :width_half]
        right_data = gray_frame_cached[:, width_half:]

        return SplitSharedFrame(
            left=NeonCameraInterface.frame_from_template(shared_frame, left_data),
            right=NeonCameraInterface.frame_from_template(shared_frame, right_data),
        )

    @staticmethod
    def frame_from_template(
        template: GrayFrameProtocol, data: npt.NDArray[np.uint8]
    ) -> GrayFrameProtocol:
        new_frame: GrayFrameProtocol = SimpleNamespace(
            gray=data,
            data_fully_received=template.data_fully_received,
            index=template.index,
            timestamp=template.timestamp,
        )  # type: ignore
        return new_frame

    def close(self):
        if self._uvc_capture is not None:
            self._uvc_capture.close()
        self._uvc_capture = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    @property
    def controls(self) -> Dict[str, Any]:
        if self._uvc_capture is None:
            return {}
        return {c.display_name: c.value for c in self._uvc_capture.controls}

    @controls.setter
    def controls(self, to_be_changed: Dict[str, Any]):
        if self._uvc_capture is None:
            return

        controls: Dict[str, Any] = {
            c.display_name: c for c in self._uvc_capture.controls
        }
        for key in set(to_be_changed) & set(controls):
            controls[key].value = to_be_changed[key]
