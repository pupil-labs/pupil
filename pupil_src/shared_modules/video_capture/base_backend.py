'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from time import time, sleep
from plugin import Plugin

import gl_utils
from pyglui import cygl
import numpy as np
import os
import av
from bisect import bisect_left as bisect

import logging
import pyaudio as pa
import itertools
from threading import Timer
logger = logging.getLogger(__name__)


class InitialisationError(Exception):
    pass


class StreamError(Exception):
    pass


class EndofVideoError(Exception):
    pass


class Base_Source(Plugin):
    """Abstract source class

    All source objects are based on `Base_Source`.

    A source object is independent of its matching manager and should be
    initialisable without it.

    Initialization is required to succeed. In case of failure of the underlying capture
    the follow properties need to be readable:

    - name
    - frame_rate
    - frame_size

    The recent_events function is allowed to not add a frame to the `events` object.

    Attributes:
        g_pool (object): Global container, see `Plugin.g_pool`
    """

    uniqueness = 'by_base_class'
    order = .0
    icon_chr = chr(0xe412)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.g_pool.capture = self
        self._recent_frame = None
        self._intrinsics = None

    def add_menu(self):
        super().add_menu()
        self.menu_icon.order = 0.2

    def recent_events(self, events):
        """Returns None

        Adds events['frame']=Frame(args)
            Frame: Object containing image and time information of the current
            source frame. See `fake_source.py` for a minimal implementation.
        """
        raise NotImplementedError()

    def gl_display(self):
        if self._recent_frame is not None:
            frame = self._recent_frame
            if frame.yuv_buffer is not None:
                self.g_pool.image_tex.update_from_yuv_buffer(frame.yuv_buffer,frame.width,frame.height)
            else:
                self.g_pool.image_tex.update_from_ndarray(frame.bgr)
            gl_utils.glFlush()
        gl_utils.make_coord_system_norm_based()
        self.g_pool.image_tex.draw()
        if not self.online:
            cygl.utils.draw_gl_texture(np.zeros((1, 1, 3), dtype=np.uint8), alpha=0.4)
        gl_utils.make_coord_system_pixel_based((self.frame_size[1], self.frame_size[0], 3))

    @property
    def name(self):
        raise NotImplementedError()

    def get_init_dict(self):
        return {}

    @property
    def frame_size(self):
        """Summary
        Returns:
            tuple: 2-element tuple containing width, height
        """
        raise NotImplementedError()

    @frame_size.setter
    def frame_size(self, new_size):
        raise NotImplementedError()

    @property
    def frame_rate(self):
        """
        Returns:
            int/float: Frame rate
        """
        raise NotImplementedError()

    @frame_rate.setter
    def frame_rate(self, new_rate):
        pass

    @property
    def jpeg_support(self):
        """
        Returns:
            bool: Source supports jpeg data
        """
        raise NotImplementedError()

    @property
    def online(self):
        """
        Returns:
            bool: Source is avaible and streaming images.
        """
        return True

    @property
    def intrinsics(self):
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, model):
        self._intrinsics = model


class Base_Manager(Plugin):
    """Abstract base class for source managers.

    Managers are plugins that enumerate and load accessible sources from
    different backends, e.g. locally USB-connected cameras.

    Attributes:
        gui_name (str): String used for manager selector labels
    """

    uniqueness = 'by_base_class'
    gui_name = 'Base Manager'
    icon_chr = chr(0xec01)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool):
        super().__init__(g_pool)

    def add_menu(self):
        super().add_menu()
        from . import manager_classes
        from pyglui import ui

        self.menu_icon.order = 0.1

        def replace_backend_manager(manager_class):
            if self.g_pool.process.startswith('eye'):
                self.g_pool.capture_manager.deinit_ui()
                self.g_pool.capture_manager.cleanup()
                self.g_pool.capture_manager = manager_class(self.g_pool)
                self.g_pool.capture_manager.init_ui()
            else:
                self.notify_all({'subject': 'start_plugin', 'name': manager_class.__name__})

        # We add the capture selection menu
        self.menu.append(ui.Selector(
                            'capture_manager',
                            setter    = replace_backend_manager,
                            getter    = lambda: self.__class__,
                            selection = manager_classes,
                            labels    = [b.gui_name for b in manager_classes],
                            label     = 'Manager'
                        ))

        # here is where you add all your menu entries.
        self.menu.label = "Backend Manager"


class Playback_Source(Base_Source):
    allowed_speeds = [.25, .5, 1., 1.5, 2., 4.]

    def __init__(self, g_pool, source_path=None, timed_playback=False, playback_speed=1., play_audio=False, *args, **kwargs):
        super().__init__(g_pool)
        self.playback_speed = playback_speed
        self.timed_playback = timed_playback
        self.time_discrepancy = 0.
        self._recent_wait_idx = -1
        self.play = True
        self.timestamps = None
        

        self.pa_stream = None
        self.audio_sync = 0.
        self.audio_delay = 0.
        self.audio_container = None
        self.audio_stream = None
        self.next_audio_frame = None
        if not play_audio or source_path is None:
            return
        audio_file = os.path.join(os.path.dirname(source_path), 'audio.mp4')
        if os.path.isfile(audio_file):
                self.audio_container = av.open(str(audio_file))
                try:
                    self.audio_stream = next(s for s in self.audio_container.streams if s.type == 'audio')
                    logger.debug("loaded audiostream: %s" % self.audio_stream)
                except StopIteration:
                    self.audio_stream = None
                    logger.debug("No audiostream found in media container")
        if self.audio_stream is not None:
            self.audio_bytes_fifo = []
            audiots_path = os.path.splitext(audio_file)[0] + '_timestamps.npy'
            try:
                self.audio_timestamps = np.load(audiots_path)
            except IOError:
                self.audio_timestamps = None
                logger.warning("Could not load audio timestamps")
            self.next_audio_frame = self._next_audio_frame()
            self.audio_fifo = av.audio.fifo.AudioFifo()
            self.audio_resampler = av.audio.resampler.AudioResampler(format=self.audio_stream.format.packed,
                                                                     layout=self.audio_stream.layout,
                                                                     rate=self.audio_stream.rate)
            self.audio_paused = False
            af0, af1 = next(self.next_audio_frame), next(self.next_audio_frame)
            self.audio_pts_rate = af1.samples
            self.seek_to_audio_frame(0)

            print("Audio file format {} chans {} rate {} framesize {} ".format(self.audio_stream.format,
                                                                               self.audio_stream.channels,
                                                                               self.audio_stream.rate,
                                                                               self.audio_stream.frame_size))

            def audio_callback(in_data, frame_count, time_info, status):
                #return (None, pa.paComplete)
                #frame = self.audio_fifo.read(frame_count)
                #samples = frame.planes[0].tobytes()
                #print("Time diff {}".format(time_info['output_buffer_dac_time'] - time_info['current_time']))
                if not self.play:
                    self.audio_paused = True
                    print("audio cb abort 1")
                    return (None, pa.paAbort)
                try:
                    samples = self.audio_bytes_fifo.pop(0)
                    return (samples, pa.paContinue)
                except IndexError:
                    self.audio_paused = True
                    print("audio cb abort 2")
                    return (None, pa.paAbort)


            try:
                self.pa = pa.PyAudio()
                self.pa_stream = self.pa.open(format=self.pa.get_format_from_width(self.audio_stream.format.bytes),
                                              channels=self.audio_stream.channels,
                                              rate=self.audio_stream.rate,
                                              frames_per_buffer=self.audio_stream.frame_size,
                                              stream_callback=audio_callback,
                                              output=True,
                                              start=False)
                print("Audio output latency: {}".format(self.pa_stream.get_output_latency()))
                self.audio_sync = self.pa_stream.get_output_latency()

            except ValueError:
                self.pa_stream = None

    def _next_audio_frame(self):
        for packet in self.audio_container.demux(self.audio_stream):
            for frame in packet.decode():
                if frame:
                    yield frame
        raise StopIteration()

    def audio_idx_to_pts(self, idx):
        return idx*self.audio_pts_rate

    def seek_to_audio_frame(self, seek_pos):
        try:
            self.audio_stream.seek(self.audio_idx_to_pts(seek_pos), mode='time')
        except av.AVError as e:
            raise FileSeekError()
        else:
            self.next_audio_frame = self._next_audio_frame()
            self.audio_bytes_fifo.clear()     

    def seek_to_frame(self, frame_idx):
        if self.audio_stream is not None:
            audio_idx = bisect(self.audio_timestamps, self.timestamps[frame_idx])
            self.seek_to_audio_frame(audio_idx)


    def get_frame_index(self):
        raise NotImplementedError()

    def seek_to_prev_frame(self):
        raise NotImplementedError()

    def get_frame(self, frame_idx=-1):
        if self.pa_stream is not None and self.play:
            samples_written = 0
            if self.playback_speed == 1.:
                if self.pa_stream.is_stopped() or self.audio_paused:
                    if frame_idx == -1:
                        frame_idx = 0
                    ts_delay = self.audio_timestamps[0] - self.timestamps[frame_idx]
                    if ts_delay > 0.:
                        delay_lat = ts_delay - self.pa_stream.get_output_latency()
                        if delay_lat > 0.:
                            self.audio_delay = delay_lat
                            self.audio_sync = 0
                        else:
                            self.audio_delay = 0
                            self.audio_sync = - delay_lat
                    else:
                        self.audio_delay = 0.
                        self.audio_sync = self.pa_stream.get_output_latency()

                    audio_idx = bisect(self.audio_timestamps, self.timestamps[frame_idx])

                    self.seek_to_audio_frame(audio_idx)

                frames_chunk = itertools.islice(self.next_audio_frame, 10)
                for audio_frame_p in frames_chunk:
                    audio_frame = self.audio_resampler.resample(audio_frame_p)
                    self.audio_bytes_fifo.append(bytes(audio_frame.planes[0]))
                    #self.audio_fifo.write(audio_frame)
                #print("AudioFIFO samples: {}".format(samples_written))
                if self.pa_stream.is_stopped() or self.audio_paused:
                    self.pa_stream.stop_stream()
                    if self.audio_delay < 0.001:
                        self.pa_stream.start_stream()
                    else:
                        def delayed_audio_start():
                            if self.pa_stream.is_stopped():
                                self.pa_stream.start_stream()
                                print("Started delayed audio")
                            self.audio_timer.cancel()

                        self.audio_timer = Timer(self.audio_delay,  delayed_audio_start)
                        self.audio_timer.start()

                    self.audio_paused = False

            elif not self.pa_stream.is_stopped():
                self.pa_stream.stop_stream()

    def wait(self, frame):
        if frame.index == self._recent_wait_idx:
            sleep(1/60)  # 60 fps on Player pause
        elif self.time_discrepancy:
            wait_time = frame.timestamp - self.time_discrepancy - time()
            wait_time /= self.playback_speed
            if 1 > wait_time > 0:
                sleep(wait_time)
        self._recent_wait_idx = frame.index
        self.time_discrepancy = frame.timestamp - time()

    def set_audio_latency(self, latency):
        if self.pa_stream is not None:
            pass


