from typing import Dict, NamedTuple, Tuple

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

    def matches_device(self, device: UVCDeviceInfo) -> bool:
        return self.matches_vid_pid(device["idVendor"], device["idProduct"])

    @classmethod
    def spec_by_name(cls) -> Dict[str, "CameraSpec"]:
        return {spec.name: spec for spec in (SCENE_CAM_SPEC, MODULE_SPEC)}


SCENE_CAM_SPEC = CameraSpec(
    name="Neon Scene Camera v1",
    vendor_id=0x0BDA,
    product_id=0x3036,
    width=1600,
    height=1200,
    fps=30,
    bandwidth_factor=1.6,
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


SCENE_CAM_INTRINSICS = Intrinsics(
    projection_matrix=(
        (892.1746128870618, 0.0, 829.7903330088201),
        (0.0, 891.4721112020742, 606.9965952706247),
        (0.0, 0.0, 1.0),
    ),
    distortion_coeffs=(
        (
            -0.13199101574152391,
            0.11064108837365579,
            0.00010404274838141136,
            -0.00019483441697480834,
            -0.002837744957163781,
            0.17125797998042083,
            0.05167573834059702,
            0.021300346544012465,
        )
    ),
)

RIGHT_EYE_CAM_INTRINSICS = Intrinsics(
    projection_matrix=(
        (140.68445787837342, 0.0, 99.42393317744813),
        (0.0, 140.67571954970256, 96.235134525304),
        (0.0, 0.0, 1.0),
    ),
    distortion_coeffs=(
        (
            0.05449484235207129,
            -0.14013187141454536,
            0.0006598061556076783,
            5.0572400552608696e-05,
            -0.6158040573125376,
            -0.048953803434398195,
            0.04521347340211147,
            -0.7004955138758611,
        )
    ),
)
LEFT_EYE_CAM_INTRINSICS = Intrinsics(
    projection_matrix=(
        (139.9144144115404, 0.0, 93.41071490844465),
        (0.0, 140.1345290475987, 95.95430565624916),
        (0.0, 0.0, 1.0),
    ),
    distortion_coeffs=(
        (
            0.0557006697189688,
            -0.13905200286485173,
            -0.0023103220308350637,
            2.777911826660397e-05,
            -0.636978924394666,
            -0.05039411576001638,
            0.044410690945337346,
            -0.7150659160991246,
        )
    ),
)

NEON_SHARED_EYE_CAM_TOPIC = "neon_backend.shared_eye_cam."
