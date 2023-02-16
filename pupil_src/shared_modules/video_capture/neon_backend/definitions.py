import sys
from typing import Dict, NamedTuple, Tuple, get_type_hints

import numpy as np
import numpy.typing as npt
from typing_extensions import Protocol, TypedDict


class UVCDeviceInfo(TypedDict):
    uid: str
    idVendor: int
    idProduct: int


class CameraSpec(NamedTuple):
    name: str
    vendor_id: int
    product_id: int
    width: int
    height: int
    fps: int
    bandwidth_factor: float

    def matches_vid_pid(self, vendor_id: int, product_id: int) -> bool:
        return self.vendor_id == vendor_id and self.product_id == product_id

    def matches_name(self, name: str) -> bool:
        return self.name == name

    def matches_device(self, device: UVCDeviceInfo) -> bool:
        return self.matches_name(device["name"])

    @classmethod
    def spec_by_name(cls) -> Dict[str, "CameraSpec"]:
        return {spec.name: spec for spec in (SCENE_CAM_SPEC, MODULE_SPEC)}


SCENE_CAM_SPEC = CameraSpec(
    name="Neon Scene Camera v1",
    vendor_id=0x0BDA,
    product_id=0x3036,
    width=1280,
    height=720,
    fps=30,
    bandwidth_factor=1.2,
)
MODULE_SPEC = CameraSpec(
    name="Neon Sensor Module v1",
    vendor_id=0x04B4,
    product_id=0x0036,
    width=384,
    height=192,
    fps=200,
    bandwidth_factor=0,
)


class GrayFrameProtocol(Protocol):
    gray: npt.NDArray[np.uint8]
    data_fully_received: bool
    index: int
    timestamp: float


class SplitSharedFrame(NamedTuple):
    left: GrayFrameProtocol
    right: GrayFrameProtocol


ProjectionMatrix = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]

DistortionCoeffs = Tuple[float, ...]


class Intrinsics(NamedTuple):
    projection_matrix: ProjectionMatrix
    distortion_coeffs: DistortionCoeffs


NEON_SHARED_EYE_FRAME_TOPIC = "neon_backend.shared_eye_frame."

_STATE_TOPIC_PREFIX = "neon_backend.shared_cam_state."
NEON_SHARED_CAM_STATE_ANNOUNCEMENT_TOPIC = _STATE_TOPIC_PREFIX + "announcement"
NEON_SHARED_CAM_STATE_CHANGE_REQUEST_TOPIC = _STATE_TOPIC_PREFIX + "change_request"
