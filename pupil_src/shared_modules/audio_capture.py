'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import av
import queue

from plugin import Plugin
from pyglui import ui
from audio import Audio_Input_Dict
from threading import Thread, Event

import platform
import logging

assert(av.__version__ >= '0.3.1')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
av.logging.set_level(av.logging.ERROR)


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
        self.queue = queue.Queue()
        self.start_capture(self.audio_src)

    def recent_events(self, events):
        audio_packets = []
        while True:
            try:
                packet = self.queue.get_nowait()
            except queue.Empty:
                break
            audio_packets.append(packet)
        events['audio_packets'] = audio_packets

    def init_gui(self):
        self.menu = ui.Growing_Menu('Audio Capture')
        self.menu.collapsed = True
        self.g_pool.sidebar.append(self.menu)

        def close():
            self.alive = False
        help_str = 'Creates events for audio input.'
        self.menu.append(ui.Button('Close', close))
        self.menu.append(ui.Info_Text(help_str))
        state_label = ui.Text_Input('running', self, label='Running:', getter=lambda: self.running.is_set())
        state_label.read_only = True
        self.menu.append(state_label)

        def audio_dev_getter():
            # fetch list of currently available
            self.audio_devices_dict = Audio_Input_Dict()
            devices = list(self.audio_devices_dict.keys())
            return devices, devices

        self.menu.append(ui.Selector('audio_src', self,
                                     selection_getter=audio_dev_getter,
                                     label='Audio Source',
                                     setter=self.start_capture))

    def get_init_dict(self):
        return {'audio_src': self.audio_src}

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def cleanup(self):
        self.running.clear()
        self.deinit_gui()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

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
                             args=(self.audio_devices_dict[audio_src], self.running))
        self.thread.start()

    def capture_thread(self, audio_src, running):
        try:
            if platform.system() == "Darwin":
                in_container = av.open('none:{}'.format(audio_src), format="avfoundation")
            elif platform.system() == "Linux":
                in_container = av.open('hw:{}'.format(audio_src), format="alsa")
            else:
                raise av.AVError('Platform does not support audio capture.')
        except av.AVError:
            running.clear()
            return

        in_stream = None
        for idx, stream in enumerate(in_container.streams):
            if stream.type == 'audio':
                in_stream = stream
                break

        if not in_stream:
            logger.warning('No audio stream found for selected device.')
            running.clear()
            return

        now = self.g_pool.get_now()
        for packet in in_container.demux(in_stream):
            try:
                packet.timestamp = (packet.pts - in_stream.start_time) * in_stream.time_base + now
                self.queue.put_nowait(packet)
            except queue.Full:
                pass  # drop packet
            if not running.is_set():
                return

        self.audio_src = 'No Audio'
        running.clear()  # in_stream stopped yielding packets
