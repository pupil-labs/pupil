'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import sys
import os
import platform
import glob
import zmq
import zmq_tools
import numpy as np
from plugin import Producer_Plugin_Base
from pyglui import ui
from time import sleep
from player_methods import correlate_data
from file_methods import load_object,save_object

import pupil_detectors  # trigger module compilation

import logging
logger = logging.getLogger(__name__)


class Empty(object):
        pass


class Pupil_Producer_Base(Producer_Plugin_Base):
    uniqueness = 'by_base_class'
    order = 0.01


class Pupil_From_Recording(Pupil_Producer_Base):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        g_pool.pupil_positions = g_pool.pupil_data['pupil_positions']
        g_pool.pupil_positions_by_frame = correlate_data(g_pool.pupil_positions, g_pool.timestamps)
        self.notify_all({'subject': 'pupil_positions_changed'})
        logger.debug('pupil positions changed')

    def get_init_dict(self):
        return {}


class Offline_Pupil_Detection(Pupil_Producer_Base):
    """docstring for Offline_Pupil_Detection"""
    session_data_version = 1

    def __init__(self, g_pool):
        super().__init__(g_pool)
        zmq_ctx = zmq.Context()
        self.data_sub = zmq_tools.Msg_Receiver(zmq_ctx, g_pool.ipc_sub_url, topics=('pupil',))

        self.data_dir = os.path.join(g_pool.rec_dir, 'offline_data')
        os.makedirs(self.data_dir , exist_ok=True)
        try:
            session_data = load_object(os.path.join(self.data_dir , 'offline_pupil_data'))
            assert session_data.get('version') != self.session_data_version
        except:
            session_data = {}
            session_data["detection_method"]='3d'
            session_data['pupil_positions'] = []
            session_data['detection_progress'] = [0.,0.]
            session_data['detection_status'] = ["unknown","unknown"]
        self.detection_method = session_data["detection_method"]
        self.pupil_positions = session_data['pupil_positions']
        self.eye_processes = [None, None]
        self.detection_progress = session_data['detection_progress']
        self.detection_status = session_data['detection_status']

        self.menu = None

        # start processes
        if self.detection_progress[0] < 100:
            self.start_eye_process(0)
        if self.detection_progress[1] < 100:
            self.start_eye_process(1)

        # either we did not start them or they failed to start (mono setup etc)
        # either way we are done and can publish
        if self.eye_processes == [None, None]:
            self.correlate_publish()

    def start_eye_process(self, eye_id):
        potential_locs = [os.path.join(self.g_pool.rec_dir, 'eye{}{}'.format(eye_id, ext)) for ext in ('.mjpeg', '.mp4')]
        existing_locs = [loc for loc in potential_locs if os.path.exists(loc)]
        timestamps_path = os.path.join(self.g_pool.rec_dir,'eye{}_timestamps.npy'.format(eye_id))

        if not existing_locs:
            logger.error("no eye video for eye '{}' found.".format(eye_id))
            self.detection_status[eye_id] = "No eye video found!"
            return
        if not os.path.exists(timestamps_path):
            logger.error("no timestamps for eye video for eye '{}' found.".format(eye_id))
            return

        video_loc = existing_locs[0]
        ts = np.load(timestamps_path)
        self.detection_progress[eye_id] = 0.
        capure_settings = 'File_Source', {
            'source_path': video_loc,
            'timestamps': ts.tolist(),
            'timed_playback': False
        }
        self.notify_all({'subject': 'eye_process.should_start', 'eye_id': eye_id,
                         'overwrite_cap_settings': capure_settings})
        eye_p = Empty()  # dummy object holding meta data
        eye_p.video_path = video_loc
        eye_p.min_ts = ts[0]
        eye_p.max_ts = ts[-1]
        self.eye_processes[eye_id] = eye_p
        self.detection_status[eye_id] = "Detecting..."

    def stop_eye_process(self, eye_id):
        self.notify_all({'subject': 'eye_process.should_stop', 'eye_id': eye_id})
        self.eye_processes[eye_id] = None

    def recent_events(self, events):
        while self.data_sub.new_data:
            topic, payload = self.data_sub.recv()
            if topic.startswith('pupil.'):
                self.pupil_positions.append(payload)
                self.update_progress(payload)
        if self.eye_processes[0] and self.detection_progress[0] == 100.:
            logger.debug("eye 0 process complete")
            self.detection_status[0] = "complete"
            self.stop_eye_process(0)
            if self.eye_processes == [None,None]:
                self.correlate_publish()
        if self.eye_processes[1] and self.detection_progress[1] == 100.:
            logger.debug("eye 1 process complete")
            self.detection_status[1] = "complete"
            self.stop_eye_process(1)
            if self.eye_processes == [None,None]:
                self.correlate_publish()

    def correlate_publish(self):
        self.g_pool.pupil_positions = self.pupil_positions
        self.g_pool.pupil_positions_by_frame = correlate_data(self.pupil_positions, self.g_pool.timestamps)
        self.notify_all({'subject': 'pupil_positions_changed'})
        logger.debug('pupil positions changed')

    def on_notify(self, notification):
        if notification['subject'] == 'eye_process.started':
            self.set_detection_mapping_mode(self.detection_method)

    def update_progress(self, pupil_position):
        eye_id = pupil_position['id']
        cur_ts = pupil_position['timestamp']
        min_ts = self.eye_processes[eye_id].min_ts
        max_ts = self.eye_processes[eye_id].max_ts
        self.detection_progress[eye_id] = 100 * (cur_ts - min_ts) / (max_ts - min_ts)

    def cleanup(self):
        self.stop_eye_process(0)
        self.stop_eye_process(1)
        # close sockets before context is terminated
        self.data_sub = None
        self.deinit_gui()

        session_data = {}
        session_data["detection_method"]= self.detection_method
        session_data['pupil_positions'] = self.pupil_positions
        session_data['detection_progress'] = self.detection_progress
        session_data['detection_status'] = self.detection_status
        save_object(session_data,os.path.join(self.data_dir,'offline_pupil_data'))

    def redetect(self):
        del self.pupil_positions[:]  # delete previously detected pupil positions
        self.g_pool.pupil_positions_by_frame = [[] for x in self.g_pool.timestamps]
        self.detection_finished_flag = False
        self.detection_progress[0] = 0.
        self.detection_progress[1] = 0.
        for eye_id in range(2):
            if self.eye_processes[eye_id] is None:
                self.start_eye_process(eye_id)
            else:
                self.notify_all({'subject': 'file_source.seek',
                                        'frame_index': 0,
                                         'source_path': self.eye_processes[eye_id].video_path})

    def set_detection_mapping_mode(self, new_mode):
        n = {'subject': 'set_detection_mapping_mode', 'mode': new_mode}
        self.notify_all(n)
        self.redetect()
        self.detection_method = new_mode

    def init_gui(self):
        self.menu = ui.Scrolling_Menu("Offline Pupil Detector", size=(220, 300))
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Selector('detection_method', self, label='Detection Method',
                                     selection=['2d', '3d'], setter=self.set_detection_mapping_mode))
        self.menu.append(ui.Button('Redetect', self.redetect))
        self.menu.append(ui.Text_Input("0",label='eye0:',getter=lambda :self.detection_status[0],
                                    setter=lambda _: _))
        progress_slider = ui.Slider('0',
                                    label='Progress Eye 0',
                                    getter=lambda :self.detection_progress[0],
                                    setter=lambda _: _)
        progress_slider.display_format = '%3.0f%%'
        self.menu.append(progress_slider)
        self.menu.append(ui.Text_Input("1",label='eye1:',getter=lambda :self.detection_status[1],
                                    setter=lambda _: _))
        progress_slider = ui.Slider('1',
                                    label='Progress Eye 1',
                                    getter=lambda :self.detection_progress[1],
                                    setter=lambda _: _)
        progress_slider.display_format = '%3.0f%%'
        self.menu.append(progress_slider)

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        return {}

