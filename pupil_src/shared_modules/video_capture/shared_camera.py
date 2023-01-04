"""

"""

import ctypes
import logging
import multiprocessing
import traceback
from dataclasses import dataclass
from multiprocessing.sharedctypes import SynchronizedBase
from multiprocessing.synchronize import Event as EventClass
from typing import NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
import zmq_tools
from typing_extensions import Literal


def start_event_loop_in_background(
    uvc_uid: str,
    timebase: SynchronizedBase[ctypes.c_double],  # mp.Value
    ipc_pub_url: str,
    ipc_sub_url: str,
    ipc_push_url: str,
    debug: bool = False,
    wait_for_process_start: bool = True,
) -> EventClass:
    process_started_event = multiprocessing.Event()
    should_stop_running_event = multiprocessing.Event()

    multiprocessing.Process(
        name="Shared Camera Process",
        target=event_loop,
        args=(
            process_started_event,
            should_stop_running_event,
            uvc_uid,
            timebase,
            ipc_pub_url,
            ipc_sub_url,
            ipc_push_url,
            debug,
        ),
    ).start()

    if wait_for_process_start:
        process_started_event.wait()

    return should_stop_running_event


def event_loop(
    process_started_event: EventClass,
    should_stop_running_event: EventClass,
    *args,
    **kwargs,
):
    with SharedCameraProcessManager(*args, **kwargs) as scpm:
        process_started_event.set()
        while not should_stop_running_event.is_set():
            scpm.publish_camera_frame()


@dataclass
class SharedCameraProcessManager:
    camera_spec: "CameraSpec"
    timebase: SynchronizedBase[ctypes.c_double]  # mp.Value
    ipc_pub_url: str
    ipc_sub_url: str
    ipc_push_url: str
    debug: bool = False

    def __enter__(self):
        self._setup_networking()
        self._setup_logging()
        self._setup_camera()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if exception_type is not None:
            logger.error(
                traceback.format_exception(
                    exception_type, exception_value, exception_traceback
                )
            )

    def publish_camera_frame(self):
        pass

    def _setup_camera(self):
        pass

    def _setup_networking(self):
        import zmq

        self.zmq_ctx = zmq.Context()
        self.ipc_pub = zmq_tools.Msg_Streamer(self.zmq_ctx, self.ipc_pub_url)
        self.notify_sub = zmq_tools.Msg_Receiver(
            self.zmq_ctx, self.ipc_sub_url, topics=("notify",)
        )

    def _setup_logging(self):
        # log setup
        logging.getLogger("OpenGL").setLevel(logging.ERROR)
        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.NOTSET)
        logger.addHandler(zmq_tools.ZMQ_handler(self.zmq_ctx, self.ipc_push_url))
        # create logger for the context of this function
        self.logger = logging.getLogger(__name__)

    def send_eye_images(
        self,
        frame,
        topic_prefix: str,
    ):
        gray_frame_cached: npt.NDArray[np.uint8] = frame.gray
        if not frame.data_fully_received:
            self.logger.debug("Frame data not fully received. Dropping.")
            return

        width_half = gray_frame_cached.shape[1] // 2
        left_frame = gray_frame_cached[:, :width_half]
        right_frame = gray_frame_cached[:, width_half:]

        self._send_image(
            image=left_frame,
            projection_matrix=self.camera_spec.projection_matrix_left,
            distortion_coeffs=self.camera_spec.distortion_coeffs_left,
            format_="gray",
            index=frame.index,
            timestamp=frame.timestamp,
            topic=topic_prefix + "eye1",
        )
        self._send_image(
            image=right_frame,
            projection_matrix=self.camera_spec.projection_matrix_right,
            distortion_coeffs=self.camera_spec.distortion_coeffs_right,
            format_="gray",
            index=frame.index,
            timestamp=frame.timestamp,
            topic=topic_prefix + "eye0",
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
                "timestamp": timestamp - self.timebase.value,
                "__raw_data__": [np.ascontiguousarray(image)],
            }
        )


ProjectionMatrix = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]

DistortionCoeffs = Tuple[float, ...]


class CameraSpec(NamedTuple):
    uid: str
    width: int
    height: int
    fps: int
    bandwidth_factor: float
    projection_matrix_left: ProjectionMatrix
    projection_matrix_right: ProjectionMatrix
    distortion_coeffs_left: DistortionCoeffs = (0.0,) * 5
    distortion_coeffs_right: DistortionCoeffs = (0.0,) * 5
