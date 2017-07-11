'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import cv2
import zmq
import numpy as np
import zmq_tools
from time import sleep
from circle_detector import find_concetric_circles
from video_capture import File_Source, EndofVideoFileError
from methods import normalize

import logging
logger = logging.getLogger(__name__)


class Empty(object):
        pass


def circle_detector(ipc_pub_url, ipc_sub_url, ipc_push_url,
                    source_path, timestamps_path, batch_size=20):

    zmq_ctx = zmq.Context()
    ipc_pub = zmq_tools.Msg_Dispatcher(zmq_ctx, ipc_push_url)
    ipc_sub = zmq_tools.Msg_Receiver(zmq_ctx, ipc_sub_url,
                                     topics=('notify.circle_detector.',))

    while True:
        # Wait until subscriptions were successfull
        ipc_pub.notify({'subject': 'circle_detector.startup'})
        if ipc_sub.socket.poll(timeout=50):
            ipc_sub.recv()
            break

    finished_successfull = False

    try:
        src = File_Source(Empty(), source_path, np.load(timestamps_path), timed_playback=False)
        frame = src.get_frame()
        logger.info('Starting calibration marker detection...')
        frame_count = src.get_frame_count()

        queue = []

        while True:
            while ipc_sub.socket.poll(timeout=0):
                topic, n = ipc_sub.recv()
                if topic == 'notify.circle_detector.should_stop':
                    reason = 'Early cancellation'
                    break

            progress = 100.*frame.index/frame_count

            markers = find_concetric_circles(frame.gray, min_ring_count=3)
            if len(markers) > 0:
                detected = True
                marker_pos = markers[0][0][0]  # first marker innermost ellipse, pos
                pos = normalize(marker_pos, (frame.width, frame.height), flip_y=True)

            else:
                detected = False
                pos = None

            if detected:
                second_ellipse = markers[0][1]
                col_slice = int(second_ellipse[0][0]-second_ellipse[1][0]/2),int(second_ellipse[0][0]+second_ellipse[1][0]/2)
                row_slice = int(second_ellipse[0][1]-second_ellipse[1][1]/2),int(second_ellipse[0][1]+second_ellipse[1][1]/2)
                marker_gray = frame.gray[slice(*row_slice), slice(*col_slice)]
                avg = cv2.mean(marker_gray)[0]
                center = marker_gray[int(second_ellipse[1][1])//2, int(second_ellipse[1][0])//2]
                rel_shade = center-avg

                ref = {}
                ref["norm_pos"] = pos
                ref["screen_pos"] = marker_pos
                ref["timestamp"] = frame.timestamp
                ref['index'] = frame.index
                if rel_shade > 30:
                    ref['type'] = 'stop_marker'
                else:
                    ref['type'] = 'calibration_marker'

                queue.append((progress, ref))
            else:
                queue.append((progress, None))

            if len(queue) > batch_size:
                # dequeue batch
                data = queue[:batch_size]
                del queue[:batch_size]
                ipc_pub.notify({'subject': 'circle_detector.progressed', 'data': data})

            frame = src.get_frame()
    except EndofVideoFileError:
        finished_successfull = True
        ipc_pub.notify({'subject': 'circle_detector.progressed', 'data': queue})
    except:
        import traceback
        reason = traceback.format_exc()

    if finished_successfull:
        ipc_pub.notify({'subject': 'circle_detector.finished'})
    else:
        ipc_pub.notify({'subject': 'Empty.exception', 'reason': reason})
    sleep(1.0)
