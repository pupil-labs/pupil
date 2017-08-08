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
import av
import queue
import numpy as np

from plugin import Plugin
from pyglui import ui
from audio import Audio_Input_Dict
from threading import Thread, Event
from scipy.interpolate import interp1d
from time import time

import platform
import logging

assert(av.__version__ >= '0.3.0')
logger = logging.getLogger(__name__)
av.logging.set_level(av.logging.ERROR)


NOT_REC_STR = 'Start a new recording to save audio.'
REC_STR = 'Saving audio to "audio.mp4".'


class Audio_Capture(Plugin):
    """docstring for Audio_Capture"""
    def __init__(self, g_pool, audio_src='No Audio'):
        super().__init__(g_pool)
        self.audio_devices_dict = Audio_Input_Dict()
        if audio_src in list(self.audio_devices_dict.keys()):
            self.audio_src = audio_src
        else:
            self.audio_src = 'No Audio'

        self.thread = None
        self.running = Event()
        self.recording = Event()
        self.recording.clear()
        self.audio_container = None
        self.audio_out_stream = None
        self.queue = queue.Queue()
        self.start_capture(self.audio_src)

    def init_gui(self):
        self.menu = ui.Growing_Menu('Audio Capture')
        self.menu.collapsed = True
        self.g_pool.sidebar.append(self.menu)

        def close():
            self.alive = False
        help_str = 'Creates events for audio input.'
        self.menu.append(ui.Button('Close', close))
        self.menu.append(ui.Info_Text(help_str))

        def audio_dev_getter():
            # fetch list of currently available
            self.audio_devices_dict = Audio_Input_Dict()
            devices = list(self.audio_devices_dict.keys())
            return devices, devices

        self.menu.append(ui.Selector('audio_src', self,
                                     selection_getter=audio_dev_getter,
                                     label='Audio Source',
                                     setter=self.start_capture))

        self.menu.append(ui.Info_Text(NOT_REC_STR))

    def get_init_dict(self):
        return {'audio_src': self.audio_src}

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def cleanup(self):
        if self.audio_container is not None:
            self.close_audio_recording()
        self.running.clear()
        self.deinit_gui()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

    def on_notify(self, notification):
        if notification['subject'] == 'recording.started':
            self.rec_dir = notification['rec_path']
            self.recording.set()
            if self.running.is_set():  # and self.audio_container is None:
                self.menu[-2].read_only = True
                del self.menu[-1]
                self.menu.append(ui.Info_Text(REC_STR))
            elif not self.running.is_set():
                logger.warning('Recording was started without an active audio capture')
        elif notification['subject'] == 'recording.stopped':
            self.recording.clear()
            self.close_audio_recording()

    def close_audio_recording(self):
        self.menu[-2].read_only = False
        del self.menu[-1]
        self.menu.append(ui.Info_Text(NOT_REC_STR))

    def write_audio_packet(self, audio_frame):
        # Test if audio outstream has been initialized
        if self.audio_out_stream is None:
            try:
                self.audio_out_stream = self.audio_container.add_stream('aac')
            except ValueError as e:
                # packet.stream codec is not supported in target container.
                logger.error('Failed to create audio stream. Aborting recording.')
                logger.debug('Reason: {}'.format(e))
                self.close_audio_recording()

        self.timestamps.append(audio_frame.timestamp)
        packet = self.audio_out_stream.encode(audio_frame)
        if packet is not None:
            self.audio_container.mux(packet)

    def start_capture(self, audio_src):

        if self.thread and self.thread.is_alive():
            if self.audio_src == audio_src:
                return  # capture is already running for our selected source
            # else stop current capture gracefully
            self.running.clear()
            logger.debug('Closing capture for "{}"'.format(self.audio_src))
            self.thread.join(timeout=1)
            self.thread = None
        if audio_src not in self.audio_devices_dict:
            logger.warning('Selected audio source is not available anymore')
            return

        self.audio_src = audio_src
        if audio_src == 'No Audio':
            return

        self.running.set()
        self.thread = Thread(target=self.capture_thread,
                             args=(self.audio_devices_dict[audio_src], self.running, self.recording))
        self.thread.start()

    def capture_thread(self, audio_src, running, recording):
        try:
            if platform.system() == "Darwin":
                in_container = av.open('none:{}'.format(audio_src), format="avfoundation")
            elif platform.system() == "Linux":
                in_container = av.open('hw:{}'.format(audio_src), format="alsa")
            else:
                raise av.AVError('Platform does not support audio capture.')
        except av.AVError as err:
            running.clear()
            self.audio_src = 'No Audio'
            logger.warning('Error starting audio capture: {}'.format(err))
            return

        in_stream = None
        try:
            in_stream = in_container.streams.audio[0]
        except IndexError:
            logger.warning('No audio stream found for selected device.')
            running.clear()
            self.audio_src = 'No Audio'
            return

        out_container = None
        out_stream = None
        timestamps = None
        out_frame_num = 0
        in_frame_size = 0

        stream_epoch = in_stream.start_time * in_stream.time_base
        uvc_clock_dif = abs(stream_epoch - self.g_pool.get_now())
        pyt_clock_dif = abs(stream_epoch - time())
        max_clock_dif = 4.

        if uvc_clock_dif > max_clock_dif and pyt_clock_dif > max_clock_dif:
            logger.error('Could not identify audio stream clock.')
            running.clear()
            self.audio_src = 'No Audio'
            return
        elif uvc_clock_dif > pyt_clock_dif:
            logger.info('Audio stream uses time.time() as clock (Δ {}s)'.format(pyt_clock_dif))
            clock_differences = self.g_pool.get_now() - time()
        else:
            logger.info('Audio stream uses uvc.get_time_monotonic() as clock')
            clock_differences = 0

        def close_recording():
            # Bind nonlocal variables, https://www.python.org/dev/peps/pep-3104/
            nonlocal out_container, out_stream, in_stream, in_frame_size, timestamps, out_frame_num
            if out_container is not None:
                packet = out_stream.encode(audio_frame)
                while packet is not None:
                    out_frame_num += 1
                    out_container.mux(packet)
                    packet = out_stream.encode(audio_frame)
                out_container.close()

                in_frame_rate = in_stream.rate
                # in_stream.frame_size does not return the correct value.
                out_frame_size = out_stream.frame_size
                out_frame_rate = out_stream.rate

                old_ts_idx = np.arange(0, len(timestamps) * in_frame_size, in_frame_size) / in_frame_rate
                new_ts_idx = np.arange(0, out_frame_num * out_frame_size, out_frame_size) / out_frame_rate
                interpolate = interp1d(old_ts_idx, timestamps, bounds_error=False, fill_value='extrapolate')
                new_ts = interpolate(new_ts_idx)

                ts_loc = os.path.join(self.rec_dir, 'audio_timestamps.npy')
                np.save(ts_loc, new_ts)
            out_container = None
            out_stream = None
            timestamps = None

        for packet in in_container.demux(in_stream):
            # ffmpeg timestamps - in_stream.startime = packte pts relative to startime
            # multiply with stream_timebase to get seconds
            # add start time of this stream in pupil time unadjusted
            # finally add pupil timebase offset to adjust for settable timebase.
            for audio_frame in packet.decode():
                timestamp = audio_frame.pts * in_stream.time_base + clock_differences - self.g_pool.timebase.value
                if recording.is_set():
                    if out_container is None:
                        rec_file = os.path.join(self.rec_dir, 'audio.mp4')
                        out_container = av.open(rec_file, 'w')
                        out_stream = out_container.add_stream('aac')
                        out_frame_num = 0
                        in_frame_size = audio_frame.samples  # set here to make sure full packet size is used
                        timestamps = []

                    timestamps.append(timestamp)
                    packet = out_stream.encode(audio_frame)
                    if packet is not None:
                        out_frame_num += 1
                        out_container.mux(packet)
                elif out_container is not None:
                    # recording stopped
                    close_recording()
            if not running.is_set():
                close_recording()
                return

        close_recording()
        self.audio_src = 'No Audio'
        running.clear()  # in_stream stopped yielding packets
