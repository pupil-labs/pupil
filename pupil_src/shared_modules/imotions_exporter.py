'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import csv
import av
import multiprocessing as mp
from glob import glob
from shutil import copy2
from fractions import Fraction

import numpy as np
from pyglui import ui
from plugin import Analysis_Plugin_Base
from video_capture import File_Source, EndofVideoError
from methods import denormalize
import background_helper as bh
import csv_utils

import logging
logger = logging.getLogger(__name__)


__version__ = 2


class Empty(object):
        pass


def export_undistorted_h264(distorted_video_loc, target_video_loc, export_range):
    yield "Converting scene video", .1
    capture = File_Source(Empty(), distorted_video_loc)
    if not capture.initialised:
        yield "Converting scene video failed", 0.
        return

    update_rate = 10
    start_time = None
    time_base = Fraction(1, 65535)
    average_fps = int(len(capture.timestamps) / (capture.timestamps[-1] - capture.timestamps[0]))

    target_container = av.open(target_video_loc, 'w')
    video_stream = target_container.add_stream('mpeg4', 1/time_base)
    video_stream.bit_rate = 150e6
    video_stream.bit_rate_tolerance = video_stream.bit_rate / 20
    video_stream.thread_count = max(1, mp.cpu_count() - 1)
    video_stream.width, video_stream.height = capture.frame_size

    av_frame = av.VideoFrame(*capture.frame_size, 'bgr24')
    av_frame.time_base = time_base

    capture.seek_to_frame(export_range[0])
    next_update_idx = export_range[0] + update_rate
    while True:
        try:
            frame = capture.get_frame()
        except EndofVideoError:
            break

        if frame.index > export_range[1]:
            break

        if start_time is None:
            start_time = frame.timestamp

        undistorted_img = capture.intrinsics.undistort(frame.img)
        av_frame.planes[0].update(undistorted_img)
        av_frame.pts = int((frame.timestamp - start_time) / time_base)

        packet = video_stream.encode(av_frame)
        if packet:
            target_container.mux(packet)

        if capture.current_frame_idx >= next_update_idx:
            progress = ((capture.current_frame_idx - export_range[0]) /
                        (export_range[1] - export_range[0])) * .9 + .1
            yield "Converting scene video", progress * 100.
            next_update_idx += update_rate

    while True:  # flush encoder
        packet = video_stream.encode()
        if packet:
            target_container.mux(packet)
        else:
            break

    target_container.close()
    capture.cleanup()
    yield "Converting scene video completed", 1. * 100.


class iMotions_Exporter(Analysis_Plugin_Base):
    '''iMotions Gaze and Video Exporter

    All files exported by this plugin are saved to a subdirectory within
    the export directory called "iMotions". The gaze data will be written
    into a file called "gaze.tlv" and the undistored scene video will be
    saved in a file called "scene.mp4".

    The gaze.tlv file is a tab-separated CSV file with the following fields:
        GazeTimeStamp: Timestamp of the gaze point, unit: seconds
        MediaTimeStamp: Timestamp of the scene frame to which the gaze point
                        was correlated to, unit: seconds
        MediaFrameIndex: Index of the scene frame to which the gaze point was
                         correlated to
        Gaze3dX: X position of the 3d gaze point (the point the subject looks
                 at) in the scene camera coordinate system
        Gaze3dY: Y position of the 3d gaze point
        Gaze3dZ: Z position of the 3d gaze point
        Gaze2dX: undistorted gaze pixel postion, X coordinate, unit: pixels
        Gaze2dX: undistorted gaze pixel postion, Y coordinate, unit: pixels
        PupilDiaLeft: Left pupil diameter, 0.0 if not available, unit: millimeters
        PupilDiaRight: Right pupil diameter, 0.0 if not available, unit: millimeters
        Confidence: Value between 0 and 1 indicating the quality of the gaze
                    datum. It depends on the confidence of the pupil detection
                    and the confidence of the 3d model. Higher values are good.
    '''
    icon_chr = 'iM'

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.export_task = None
        self.status = 'Not exporting'
        self.progress = 0.
        self.output = 'Not set yet'
        logger.info('iMotions Exporter has been launched.')

    def on_notify(self, notification):
        if notification['subject'] == "should_export":
            self.export_data(notification['range'], notification['export_dir'])

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'iMotions Exporter'
        self.menu.append(ui.Text_Input('status', self, label='Status', setter=lambda _: None))
        self.menu.append(ui.Text_Input('output', self, label='Last export', setter=lambda _: None))
        self.menu.append(ui.Slider('progress', self, label='Progress'))
        self.menu[-1].read_only = True
        self.menu[-1].display_format = '%.0f%%'
        self.menu.append(ui.Button('Cancel export', self.cancel))

    def cancel(self):
        if self.export_task:
            self.export_task.cancel()

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        self.cancel()

    def export_data(self, export_range, export_dir):
        rec_start = self.get_recording_start_date()
        im_dir = os.path.join(export_dir, 'iMotions_{}'.format(rec_start))
        os.makedirs(im_dir, exist_ok=True)
        user_warned_3d_only = False
        self.output = im_dir
        logger.info('Exporting to {}'.format(im_dir))

        if self.export_task:
            self.export_task.cancel()

        distorted_video_loc = [f for f in glob(os.path.join(self.g_pool.rec_dir, "world.*"))
                               if os.path.splitext(f)[-1] in ('.mp4', '.mkv', '.avi', '.mjpeg')][0]
        target_video_loc = os.path.join(im_dir, 'scene.mp4')
        generator_args = (distorted_video_loc, target_video_loc, export_range)
        self.export_task = bh.Task_Proxy('iMotions Video Export', export_undistorted_h264,
                                         args=generator_args)

        info_src = os.path.join(self.g_pool.rec_dir, 'info.csv')
        info_dest = os.path.join(im_dir, 'iMotions_info.csv')
        copy2(info_src, info_dest)  # copy info.csv file

        with open(os.path.join(im_dir, 'gaze.tlv'), 'w', encoding='utf-8', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')

            csv_writer.writerow(('GazeTimeStamp',
                                 'MediaTimeStamp',
                                 'MediaFrameIndex',
                                 'Gaze3dX',
                                 'Gaze3dY',
                                 'Gaze3dZ',
                                 'Gaze2dX',
                                 'Gaze2dY',
                                 'PupilDiaLeft',
                                 'PupilDiaRight',
                                 'Confidence'))

            for media_idx in range(*export_range):
                media_timestamp = self.g_pool.timestamps[media_idx]
                for g in self.g_pool.gaze_positions_by_frame[media_idx]:
                    try:
                        pupil_dia = {}
                        for p in g['base_data']:
                            pupil_dia[p['id']] = p['diameter_3d']

                        pixel_pos = denormalize(g['norm_pos'], self.g_pool.capture.frame_size, flip_y=True)
                        undistorted3d = self.g_pool.capture.intrinsics.unprojectPoints(pixel_pos)
                        undistorted2d = self.g_pool.capture.intrinsics.projectPoints(undistorted3d, use_distortion=False)

                        data = (g['timestamp'],
                                media_timestamp,
                                media_idx - export_range[0],
                                *g['gaze_point_3d'],  # Gaze3dX/Y/Z
                                *undistorted2d.flat,  # Gaze2dX/Y
                                pupil_dia.get(1, 0.),  # PupilDiaLeft
                                pupil_dia.get(0, 0.),  # PupilDiaRight
                                g['confidence'])  # Confidence
                    except KeyError:
                        if not user_warned_3d_only:
                            logger.error('Currently, the iMotions export only supports 3d gaze data')
                            user_warned_3d_only = True
                        continue
                    csv_writer.writerow(data)

    def recent_events(self, events):
        if self.export_task:
            recent = [d for d in self.export_task.fetch()]
            if recent:
                self.status, self.progress = recent[-1]
            if self.export_task.canceled:
                self.status = 'Export has been canceled'
                self.progress = 0.

    def gl_display(self):
        self.menu_icon.indicator_stop = self.progress / 100.

    def get_recording_start_date(self):
        csv_loc = os.path.join(self.g_pool.rec_dir, 'info.csv')
        with open(csv_loc, 'r', encoding='utf-8') as csvfile:
            rec_info = csv_utils.read_key_value_file(csvfile)
            date = rec_info['Start Date'].replace('.', '_').replace(':', '_')
            time = rec_info['Start Time'].replace(':', '_')
        return '{}_{}'.format(date, time)
