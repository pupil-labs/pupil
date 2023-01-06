import ctypes
import logging
import multiprocessing
import pathlib
import traceback
from multiprocessing.sharedctypes import SynchronizedBase
from multiprocessing.synchronize import Event as EventClass
from types import TracebackType
from typing import NamedTuple, Optional, Tuple, Type

import click
import numpy as np
import numpy.typing as npt
from typing_extensions import Literal, Protocol


class BackgroundCameraSharingManager:
    def __init__(
        self,
        timebase: "SynchronizedBase[ctypes.c_double]",  # mp.Value
        ipc_pub_url: str,
        ipc_sub_url: str,
        ipc_push_url: str,
        debug: bool = False,
        wait_for_process_start: bool = True,
    ):
        process_started_event = multiprocessing.Event()
        self.should_stop_running_event = multiprocessing.Event()

        self._background_process = multiprocessing.Process(
            name="Shared Camera Process",
            target=self._event_loop,
            args=(
                process_started_event,
                self.should_stop_running_event,
                timebase,
                ipc_pub_url,
                ipc_sub_url,
                ipc_push_url,
                debug,
            ),
        )
        self._background_process.start()

        if wait_for_process_start:
            process_started_event.wait()

    def stop(self):
        self.should_stop_running_event.set()
        self._background_process.join(timeout=5.0)
        if self._background_process.exitcode is None:
            logging.getLogger(__name__ + ".foreground").warning(
                "Background process could not be terminated"
            )

    @staticmethod
    def _event_loop(
        process_started_event: EventClass,
        should_stop_running_event: EventClass,
        ipc_pub_url: str,
        ipc_sub_url: str,
        ipc_push_url: str,
        debug: bool = False,
    ):
        with NetworkInterface(
            topic_prefix="shared_camera.",
            ipc_pub_url=ipc_pub_url,
            ipc_sub_url=ipc_sub_url,
            ipc_push_url=ipc_push_url,
            debug=debug,
        ) as network:
            process_started_event.set()
            while not should_stop_running_event.is_set():
                pass


class CameraSpec(NamedTuple):
    vendor_id: int
    product_id: int
    width: int
    height: int
    fps: int
    bandwidth_factor: float


SCENE_CAM_SPEC = CameraSpec(
    vendor_id=0x0BDA,
    product_id=0x3036,
    width=1600,
    height=1200,
    fps=30,
    bandwidth_factor=1.6,
)
MODULE_SPEC = CameraSpec(
    vendor_id=0x04B4,
    product_id=0x0036,
    width=384,
    height=192,
    fps=200,
    bandwidth_factor=0,
)

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


class ModuleCameraInterface:
    def __init__(self, camera: CameraSpec, camera_reinit_timeout: float = 3) -> None:
        import uvc

        self.spec = camera

        self.logger = logging.getLogger(__name__ + ".camera")
        self._uvc_capture = None

        try:
            self._uvc_capture = self.find_connected_device(self.spec)
        except Exception as err:
            self.logger.warning(f"Camera init failed with {type(err).__name__}: {err}")

        self.timestamp_fn = uvc.get_time_monotonic
        self.last_frame_timestamp: float = self.timestamp_fn()

        self.camera_reinit_timeout = camera_reinit_timeout

    def __str__(self) -> str:
        return f"<{type(self).__name__} connected={self._uvc_capture is not None}>"

    @staticmethod
    def find_connected_device(camera: CameraSpec) -> Optional["uvc.Capture"]:
        import uvc

        logging.debug(f"Searching {camera}...")
        for device in uvc.device_list():
            if (
                device["idVendor"] == camera.vendor_id
                and device["idProduct"] == camera.product_id
            ):
                logging.debug(f"Found match by vendor/product id match")
                capture = uvc.Capture(device["uid"])
                capture.bandwidth_factor = camera.bandwidth_factor
                for mode in capture.available_modes:
                    if (mode.width, mode.height, mode.fps) == (
                        camera.width,
                        camera.height,
                        camera.fps,
                    ):
                        capture.frame_mode = mode
                        return capture
                else:
                    logging.warning(
                        f"None of the available modes matched: {capture.available_modes}"
                    )
                capture.close()
        else:
            raise OSError(
                f"No matching camera with vendor_id={camera.vendor_id:x} and "
                f"product_id={camera.product_id:x} found"
            )

    def get_frame(self) -> Optional["GrayFrameProtocol"]:

        try:
            if self._uvc_capture is None:
                raise OSError("No camera initialized")

            frame = self._uvc_capture.get_frame(timeout=0.01)
            frame.timestamp = self.last_frame_timestamp = self.timestamp_fn()
            return frame
        except (TimeoutError, OSError):
            if (
                self.timestamp_fn() - self.last_frame_timestamp
                > self.camera_reinit_timeout
            ):
                self.logger.info(
                    f"{self.camera_reinit_timeout} seconds since last frame or camera "
                    "reinit. Attempting reinit..."
                )
                self.close()
                try:
                    self._uvc_capture = self.find_connected_device(self.spec)
                except Exception as err:
                    self.logger.warning(
                        f"Camera reinit failed with {type(err).__name__}: {err}"
                    )
                self.last_frame_timestamp = self.timestamp_fn()

    def close(self):
        if self._uvc_capture is not None:
            self._uvc_capture.close()
        self._uvc_capture = None
        self._pyusb_device = None

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


class NetworkInterface:
    def __init__(
        self,
        topic_prefix: str,
        ipc_pub_url: str,
        ipc_sub_url: str,
        ipc_push_url: str,
        debug: bool = False,
        setup_zmq_handler: bool = True,
    ):
        self.topic_prefix = topic_prefix
        self._setup_networking(ipc_pub_url=ipc_pub_url, ipc_sub_url=ipc_sub_url)
        self._setup_logging(
            ipc_push_url=ipc_push_url, debug=debug, setup_zmq_handler=setup_zmq_handler
        )

    def __enter__(self):
        return self

    def __exit__(
        self,
        exception_type: Type[BaseException],
        exception_value: BaseException,
        exception_traceback: TracebackType,
    ):
        if exception_type is not None:
            self.logger.error(
                traceback.format_exception(
                    exception_type, exception_value, exception_traceback
                )
            )
        return

    def _setup_networking(self, ipc_pub_url: str, ipc_sub_url: str):
        import zmq
        import zmq_tools

        self.zmq_ctx = zmq.Context()
        self.ipc_pub = zmq_tools.Msg_Streamer(self.zmq_ctx, ipc_pub_url)
        self.notify_sub = zmq_tools.Msg_Receiver(
            self.zmq_ctx, ipc_sub_url, topics=("notify",)
        )

    def _setup_logging(
        self, ipc_push_url: str, debug: bool = False, setup_zmq_handler: bool = True
    ):
        import zmq_tools

        # log setup
        logging.getLogger("OpenGL").setLevel(logging.ERROR)
        logger = logging.getLogger()
        if setup_zmq_handler:
            logger.handlers = []
            logger.setLevel(logging.NOTSET)
            logger.addHandler(zmq_tools.ZMQ_handler(self.zmq_ctx, ipc_push_url))
        # create logger for the context of this function
        self.logger = logging.getLogger(__name__ + ".background")

    def send_eye_images(
        self,
        shared_frame: "GrayFrameProtocol",
        intrinsics_left: Intrinsics = LEFT_EYE_CAM_INTRINSICS,
        intrinsics_right: Intrinsics = RIGHT_EYE_CAM_INTRINSICS,
        timestamp_offset: float = 0.0,
    ):
        gray_frame_cached = shared_frame.gray
        if not shared_frame.data_fully_received:
            self.logger.debug("Frame data not fully received. Dropping.")
            return

        width_half = gray_frame_cached.shape[1] // 2
        left_frame = gray_frame_cached[:, :width_half]
        right_frame = gray_frame_cached[:, width_half:]

        self._send_image(
            image=left_frame,
            projection_matrix=intrinsics_left.projection_matrix,
            distortion_coeffs=intrinsics_left.distortion_coeffs,
            format_="gray",
            index=shared_frame.index,
            timestamp=shared_frame.timestamp - timestamp_offset,
            topic=self.topic_prefix + "eye1",
        )
        self._send_image(
            image=right_frame,
            projection_matrix=intrinsics_right.projection_matrix,
            distortion_coeffs=intrinsics_right.distortion_coeffs,
            format_="gray",
            index=shared_frame.index,
            timestamp=shared_frame.timestamp - timestamp_offset,
            topic=self.topic_prefix + "eye0",
        )

    def _send_image(
        self,
        image: npt.NDArray[np.uint8],
        projection_matrix: "ProjectionMatrix",
        distortion_coeffs: "DistortionCoeffs",
        format_: Literal["bgr", "gray"],
        index: int,
        timestamp: float,
        topic: str,
    ):
        height, width, *_ = image.shape
        self.ipc_pub.send(
            {
                "format": format_,
                "projection_matrix": projection_matrix,
                "distortion_coeffs": distortion_coeffs,
                "topic": topic,
                "width": width,
                "height": height,
                "index": index,
                "timestamp": timestamp,
                "__raw_data__": [np.ascontiguousarray(image)],
            }
        )


class GrayFrameProtocol(Protocol):
    gray: npt.NDArray[np.uint8]
    data_fully_received: bool
    index: int
    timestamp: float


@click.group
def test():
    pass


@test.command
def camera_interface():
    import contextlib
    import time

    import cv2

    camera = ModuleCameraInterface()
    with contextlib.suppress(KeyboardInterrupt):
        while True:
            frame = camera.get_frame()
            if frame is not None:
                cv2.imshow("Frame Preview", frame.gray)
                cv2.waitKey(5)
            else:
                time.sleep(0.005)


@test.command
@click.argument("topic_prefix", default="shared_camera_test.")
@click.argument("connection", default="tcp://127.0.0.1:{port:d}")
@click.argument(
    "shared_modules_loc",
    default=pathlib.Path(__file__).parent.parent.parent,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
)
def network_interface(
    topic_prefix: str, connection: str, shared_modules_loc: pathlib.Path
):
    import contextlib
    import sys
    import time
    from types import SimpleNamespace

    import msgpack
    import zmq

    sys.path.append(str(shared_modules_loc))

    req = zmq.Context.instance().socket(zmq.REQ)
    req.connect(connection.format(port=50020))

    req.send_string("PUB_PORT")
    pub_port = int(req.recv_string())
    push_port = pub_port  # Pupil Remote does not expose a dedicated PUSH_PORT

    req.send_string("SUB_PORT")
    sub_port = int(req.recv_string())
    for eye_id in range(2):
        req.send_string("notify.start_eye_plugin", flags=zmq.SNDMORE)
        req.send(
            msgpack.packb(
                {
                    "topic": "notify.start_eye_plugin",
                    "subject": "start_eye_plugin",
                    "target": f"eye{eye_id}",
                    "name": "HMD_Streaming_Source",
                    "args": {
                        "topics": (topic_prefix + f"eye{eye_id}",),
                        "flip_preview": False,
                        "menu_name": f"Neon {['right', 'left'][eye_id]} eye camera",
                    },
                },
                use_bin_type=True,
            )
        )
        req.recv_string()

    with (
        NetworkInterface(
            topic_prefix=topic_prefix,
            ipc_pub_url=connection.format(port=pub_port),
            ipc_push_url=connection.format(port=push_port),
            ipc_sub_url=connection.format(port=sub_port),
            setup_zmq_handler=False,
        ) as network,
        ModuleCameraInterface(MODULE_SPEC) as camera,
        contextlib.suppress(KeyboardInterrupt),
    ):
        while True:
            frame = camera.get_frame()
            if frame is not None:
                network.send_eye_images(frame)
            else:
                time.sleep(1 / 200)


if __name__ == "__main__":
    test()
