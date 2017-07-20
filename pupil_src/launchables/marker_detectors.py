'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''


class Empty(object):
        pass


def circle_detector(ipc_push_url, pair_url,
                    source_path, timestamps_path, batch_size=20):

    # ipc setup
    import zmq
    import zmq_tools
    zmq_ctx = zmq.Context()
    process_pipe = zmq_tools.Msg_Pair_Client(zmq_ctx, pair_url)

    # logging setup
    import logging
    logging.getLogger("OpenGL").setLevel(logging.ERROR)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(zmq_tools.ZMQ_handler(zmq_ctx, ipc_push_url))
    # create logger for the context of this function
    logger = logging.getLogger(__name__)

    # imports
    import cv2
    import numpy as np
    from time import sleep
    from circle_detector import find_concetric_circles
    from video_capture import File_Source, EndofVideoFileError
    from methods import normalize

    try:
        src = File_Source(Empty(), source_path, np.load(timestamps_path), timed_playback=False)
        frame = src.get_frame()
        logger.info('Starting calibration marker detection...')
        frame_count = src.get_frame_count()

        queue = []

        while True:
            while process_pipe.new_data:
                topic, n = process_pipe.recv()
                if topic == 'terminate':
                    process_pipe.send(topic='exception', payload={"reason": "User terminated."})
                    logger.debug("Process terminated")
                    sleep(1.0)
                    return

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
                col_slice = int(second_ellipse[0][0]-second_ellipse[1][0]/2), int(second_ellipse[0][0]+second_ellipse[1][0]/2)
                row_slice = int(second_ellipse[0][1]-second_ellipse[1][1]/2), int(second_ellipse[0][1]+second_ellipse[1][1]/2)
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
                process_pipe.send(topic='progress', payload={'data': data})

            frame = src.get_frame()

    except EndofVideoFileError:
        process_pipe.send(topic='progress', payload={'data': queue})
        process_pipe.send(topic='finished', payload={})
        logger.debug("Process finished")

    except:
        import traceback
        process_pipe.send(topic='exception', payload={'reason': traceback.format_exc()})
        logger.debug("Process raised Exception")

    sleep(1.0)
