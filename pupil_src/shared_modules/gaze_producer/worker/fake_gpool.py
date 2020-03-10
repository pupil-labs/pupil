import logging
from time import time
from types import SimpleNamespace


logger = logging.getLogger(__name__)


class FakeGPool:

    @staticmethod
    def from_g_pool(g_pool):
        return FakeGPool(
            rec_dir=g_pool.rec_dir,
            min_calibration_confidence=g_pool.min_calibration_confidence,
            frame_size=g_pool.capture.frame_size,
            intrinsics=g_pool.capture.intrinsics,
        )

    def __init__(self, frame_size, intrinsics, rec_dir, min_calibration_confidence):
        cap = SimpleNamespace()
        cap.frame_size = frame_size
        cap.intrinsics = intrinsics

        self.capture = cap
        self.get_timestamp = time
        self.min_calibration_confidence = min_calibration_confidence
        self.rec_dir = rec_dir
        self.app = "player"
        self.ipc_pub = FakeIPC()


class FakeIPC:
    @staticmethod
    def notify(notification, *args, **kwargs):
        name = notification.get("subject", None) or notification.get("topic", None)
        logger.debug(
            f'Received background notification "{name}"; it will be ignored.'
        )
