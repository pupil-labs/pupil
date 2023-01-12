import logging
import traceback
from types import TracebackType
from typing import Any, Dict, Iterator, Tuple, Type

import numpy as np
import numpy.typing as npt
import zmq
from typing_extensions import Literal

from .definitions import (
    NEON_SHARED_CAM_STATE_ANNOUNCEMENT_TOPIC,
    DistortionCoeffs,
    GrayFrameProtocol,
    Intrinsics,
    ProjectionMatrix,
)


class NetworkInterface:
    def __init__(
        self,
        topic_prefix: str,
        ipc_pub_url: str,
        ipc_sub_url: str,
        ipc_push_url: str,
        setup_zmq_handler: bool = True,
    ):
        self.topic_prefix = topic_prefix
        self.num_subscribers = 0
        self._setup_networking(
            ipc_pub_url=ipc_pub_url, ipc_sub_url=ipc_sub_url, ipc_push_url=ipc_push_url
        )
        self._setup_logging(
            ipc_push_url=ipc_push_url, setup_zmq_handler=setup_zmq_handler
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
                "".join(
                    traceback.format_exception(
                        exception_type, exception_value, exception_traceback
                    )
                )
            )
        return True

    def _setup_networking(self, ipc_pub_url: str, ipc_sub_url: str, ipc_push_url: str):
        import zmq
        import zmq_tools

        self.zmq_ctx = zmq.Context()
        self.num_frame_subscribers = 0
        self.ipc_pub = zmq_tools.Msg_Streamer(
            self.zmq_ctx, ipc_pub_url, socket_type=zmq.XPUB
        )
        self.notify_sub = zmq_tools.Msg_Receiver(
            self.zmq_ctx, ipc_sub_url, topics=("notify",)
        )
        self.notify_push = zmq_tools.Msg_Dispatcher(self.zmq_ctx, ipc_push_url)

    def _setup_logging(self, ipc_push_url: str, setup_zmq_handler: bool = True):
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

    def split_and_send_eye_images(
        self,
        shared_frame: "GrayFrameProtocol",
        intrinsics_left: Intrinsics,
        intrinsics_right: Intrinsics,
        timestamp_offset: float = 0.0,
    ):
        gray_frame_cached = shared_frame.gray
        if not shared_frame.data_fully_received:
            self.logger.debug("Frame data not fully received. Dropping.")
            return

        width_half = gray_frame_cached.shape[1] // 2
        left_frame = gray_frame_cached[:, :width_half]
        right_frame = gray_frame_cached[:, width_half:]

        self.send_image(
            image=left_frame,
            projection_matrix=intrinsics_left.projection_matrix,
            distortion_coeffs=intrinsics_left.distortion_coeffs,
            format_="gray",
            index=shared_frame.index,
            timestamp=shared_frame.timestamp - timestamp_offset,
            topic=self.topic_prefix + "eye1",
        )
        self.send_image(
            image=right_frame,
            projection_matrix=intrinsics_right.projection_matrix,
            distortion_coeffs=intrinsics_right.distortion_coeffs,
            format_="gray",
            index=shared_frame.index,
            timestamp=shared_frame.timestamp - timestamp_offset,
            topic=self.topic_prefix + "eye0",
        )

    def send_eye_frame(
        self,
        frame: GrayFrameProtocol,
        intrinsics: Intrinsics,
        *,
        eye_id: Literal[0, 1],
    ):
        return self.send_image(
            frame.gray,
            intrinsics.projection_matrix,
            intrinsics.distortion_coeffs,
            format_="gray",
            index=frame.index,
            timestamp=frame.timestamp,
            topic=self.topic_prefix + f"eye{eye_id}",
        )

    def send_image(
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

    def process_subscriptions(self):
        while self.ipc_pub.socket.get(zmq.EVENTS) & zmq.POLLIN:
            subscription, *_ = self.ipc_pub.socket.recv_multipart()
            if subscription.startswith(b"\x01" + self.topic_prefix.encode()):
                self.num_subscribers += 1
            elif subscription.startswith(b"\x00" + self.topic_prefix.encode()):
                self.num_subscribers = max(self.num_subscribers - 1, 0)

    def process_notifications(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        while self.notify_sub.socket.get(zmq.EVENTS) & zmq.POLLIN:
            yield self.notify_sub.recv()

    def announce_camera_state(self, state: Dict[str, Any]):
        notification = {
            "subject": NEON_SHARED_CAM_STATE_ANNOUNCEMENT_TOPIC,
            "connected": bool(state),
            **state,
        }
        self.logger.debug(f"Announcing {notification}")
        self.notify_push.notify(notification)
