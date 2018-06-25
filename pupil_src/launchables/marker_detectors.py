'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''


class Empty(object):
        pass


def circle_detector(ipc_push_url, pair_url, source_path, batch_size=20):

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
    from time import sleep
    from video_capture import init_playback_source, EndofVideoError
    from circle_detector import CircleTracker

    try:
        src = init_playback_source(Empty(), source_path, timing=None)

        frame = src.get_frame()
        logger.info('Starting calibration marker detection...')
        frame_count = src.get_frame_count()

        queue = []
        circle_tracker = CircleTracker()

        while True:
            while process_pipe.new_data:
                topic, n = process_pipe.recv()
                if topic == 'terminate':
                    process_pipe.send(topic='exception', payload={"reason": "User terminated."})
                    logger.debug("Process terminated")
                    sleep(1.0)
                    return

            progress = 100.*frame.index/frame_count

            markers = [m for m in circle_tracker.update(frame.gray) if m['marker_type'] == 'Ref']

            if len(markers):
                ref = {"norm_pos": markers[0]['norm_pos'],
                       "screen_pos": markers[0]['img_pos'],
                       "timestamp": frame.timestamp,
                       'index_range': tuple(range(frame.index - 5, frame.index + 5)),
                       'index': frame.index}
                queue.append((progress, ref))
            else:
                queue.append((progress, None))

            if len(queue) > batch_size:
                # dequeue batch
                data = queue[:batch_size]
                del queue[:batch_size]
                process_pipe.send(topic='progress', payload={'data': data})

            frame = src.get_frame()

    except EndofVideoError:
        process_pipe.send(topic='progress', payload={'data': queue})
        process_pipe.send(topic='finished', payload={})
        logger.debug("Process finished")

    except Exception:
        import traceback
        process_pipe.send(topic='exception', payload={'reason': traceback.format_exc()})
        logger.debug("Process raised Exception")

    sleep(1.0)
