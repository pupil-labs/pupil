import contextlib
import ctypes
import logging
import multiprocessing
import time
from multiprocessing.sharedctypes import SynchronizedBase
from multiprocessing.synchronize import Event as EventClass
from typing import Optional

import camera_models as cm

from .camera import NeonCameraInterface
from .definitions import (
    MODULE_SPEC,
    NEON_SHARED_CAM_STATE_CHANGE_REQUEST_TOPIC,
    Intrinsics,
)
from .network import NetworkInterface


class BackgroundCameraSharingManager:
    def __init__(
        self,
        timebase: "SynchronizedBase[ctypes.c_double]",  # mp.Value
        user_dir: str,
        ipc_pub_url: str,
        ipc_sub_url: str,
        ipc_push_url: str,
        topic_prefix: str,
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
                user_dir,
                ipc_pub_url,
                ipc_sub_url,
                ipc_push_url,
                topic_prefix,
            ),
        )
        self._background_process.start()

        if wait_for_process_start:
            process_started_event.wait()

    def stop(self):
        self.should_stop_running_event.set()
        self._background_process.join(timeout=5.0)
        if self.is_running:
            logging.getLogger(__name__ + ".foreground").debug(
                "Background process did not terminate gracefully. Forcing termination!"
            )
            self._background_process.terminate()

    @property
    def is_running(self) -> bool:
        return self._background_process.exitcode is None

    @staticmethod
    def _event_loop(
        process_started_event: EventClass,
        should_stop_running_event: EventClass,
        timebase: "SynchronizedBase[ctypes.c_double]",  # mp.Value
        user_dir: str,
        ipc_pub_url: str,
        ipc_sub_url: str,
        ipc_push_url: str,
        topic_prefix: str = "shared_camera.",
    ):
        with (
            NetworkInterface(
                topic_prefix=topic_prefix,
                ipc_pub_url=ipc_pub_url,
                ipc_sub_url=ipc_sub_url,
                ipc_push_url=ipc_push_url,
            ) as network,
            contextlib.suppress(KeyboardInterrupt),
        ):
            process_started_event.set()

            camera_model = cm.Camera_Model.from_file(
                user_dir, MODULE_SPEC.name, (MODULE_SPEC.width, MODULE_SPEC.height)
            )
            intrinsics = Intrinsics(
                projection_matrix=camera_model.K.tolist(),
                distortion_coeffs=camera_model.D.tolist(),
            )
            camera: Optional[NeonCameraInterface] = None

            while not should_stop_running_event.is_set():
                network.process_subscriptions()
                if network.num_subscribers > 0 and camera is None:
                    network.logger.debug("New subscriber(s) - start sharing camera")
                    camera = NeonCameraInterface(MODULE_SPEC)
                    last_status_update = time.perf_counter()
                    first_update = last_status_update
                    num_frames_recv = 0
                    num_frames_forwarded = 0
                    network.announce_camera_state(camera.controls)
                elif network.num_subscribers == 0 and camera is not None:
                    camera.close()
                    camera = None
                    network.logger.debug(
                        "No more subscriber(s) - stopped sharing camera"
                    )
                    network.announce_camera_state({})

                for topic, notification in network.process_notifications():
                    if (
                        notification["subject"]
                        == NEON_SHARED_CAM_STATE_CHANGE_REQUEST_TOPIC
                    ):
                        network.logger.debug(f"Received {notification}")
                        if camera is not None:
                            camera.controls = notification
                            network.announce_camera_state(camera.controls)
                        else:
                            network.announce_camera_state({})
                    if notification["subject"] == "shared_camera.restart":
                        network.logger.debug("Requested restart")
                        if camera is not None:
                            camera.close()
                            network.announce_camera_state({})
                            network.logger.debug("Closed")
                        camera = None
                        time.sleep(1.0)
                        network.logger.debug("Restarting")

                if camera is None:
                    time.sleep(0.5)
                    continue

                frame = camera.get_shared_frame(0.5)
                if frame is not None:
                    num_frames_recv += 1
                    frame.timestamp -= timebase.value
                split_frames = camera.split_shared_frame(frame)
                if split_frames is not None:
                    num_frames_forwarded += 1
                    network.send_eye_frame(split_frames.right, intrinsics, eye_id=0)
                    network.send_eye_frame(split_frames.left, intrinsics, eye_id=1)
                now = time.perf_counter()
                if now - last_status_update > 5.0:
                    network.announce_camera_state(camera.controls)
                    total_time = now - first_update
                    fps = num_frames_forwarded / total_time

                    network.logger.debug(
                        f"{num_frames_recv=} {num_frames_forwarded=} in {total_time=} "
                        f"seconds (~ {fps:.0f} FPS)"
                    )
                    last_status_update = now
