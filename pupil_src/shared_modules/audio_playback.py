"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections
import itertools
import logging
from bisect import bisect_left as bisect
from threading import Timer
from time import monotonic

import av
import av.filter
import numpy as np
import pyaudio as pa
import pyglui.cygl.utils as pyglui_utils
from pyglui import ui

import gl_utils
from audio_utils import Audio_Viz_Transform, NoAudioLoadedError, load_audio
from plugin import System_Plugin_Base
from version_utils import parse_version


assert parse_version(av.__version__) >= parse_version("0.4.4")


logger = logging.getLogger(__name__)
logger.setLevel(logger.DEBUG)

# av.logging.set_level(av.logging.DEBUG)
# logging.getLogger('libav').setLevel(logging.DEBUG)

viz_color = pyglui_utils.RGBA(0.9844, 0.5938, 0.4023, 1.0)


class FileSeekError(Exception):
    pass


class Audio_Playback(System_Plugin_Base):
    """Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites not between
    """

    icon_chr = chr(0xE050)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self.play = False
        self.pa_stream = None
        self.audio_sync = 0.0
        self.audio_delay = 0.0
        self.audio_timer = None
        self.audio_frame_iterator = None
        self.audio_start_pts = 0

        # debug flag. Only set if timestamp consistency should be checked
        self.should_check_ts_consistency = False

        self.req_audio_volume = 1.0
        self.current_audio_volume = 1.0
        self.req_buffer_size_secs = 0.5
        self.audio_viz_trans = None
        self.audio_bytes_fifo = collections.deque()

        try:
            self.audio_all = load_audio(self.g_pool.rec_dir)
        except NoAudioLoadedError:
            return

        self.calculate_audio_bounds()

        self.filter_graph = None
        self.filter_graph_list = None
        self.pa = pa.PyAudio()

        self._setup_input_audio_part(0)

        self._setup_output_audio()
        self._setup_audio_vis()

    def check_audio_part_setup(self):
        part_idx = self.audio_part_idx_from_playbacktime()
        if part_idx > -1 and part_idx != self.current_audio_part_idx:
            self._setup_input_audio_part(part_idx)

    def calculate_audio_bounds(self):
        audio_part_boundaries = (audio.timestamps[[0, -1]] for audio in self.audio_all)
        audio_part_boundaries = itertools.chain.from_iterable(audio_part_boundaries)
        self.audio_bounds = np.fromiter(audio_part_boundaries, dtype=float)

    def audio_part_idx_from_playbacktime(self):
        pbt = self.g_pool.seek_control.current_playback_time
        bound_idx = bisect(self.audio_bounds, pbt)
        if bound_idx % 2 == 0:
            return -1  # pbt is between audio parts
        else:
            part_idx = bound_idx // 2
            return part_idx

    def _setup_input_audio_part(self, part_idx):
        self.current_audio_part_idx = part_idx
        self.audio = self.audio_all[part_idx]
        self.audio_bytes_fifo.clear()

        self.audio_frame_iterator = self.get_audio_frame_iterator()
        self.audio_resampler = av.audio.resampler.AudioResampler(
            format=self.audio.stream.format.packed,
            layout=self.audio.stream.layout,
            rate=self.audio.stream.rate,
        )
        self.audio_paused = False

        self.audio.stream.seek(0)
        first_frame = next(self.audio_frame_iterator)
        self.audio_pts_rate = first_frame.samples
        self.audio_start_pts = first_frame.pts

        logger.debug(
            "audio_pts_rate = {} start_pts = {}".format(
                self.audio_pts_rate, self.audio_start_pts
            )
        )
        self.check_ts_consistency(reference_frame=first_frame)
        self.seek_to_audio_frame(0)

        logger.debug(
            "Audio file format {} chans {} rate {} framesize {}".format(
                self.audio.stream.format.name,
                self.audio.stream.channels,
                self.audio.stream.rate,
                self.audio.stream.frame_size,
            )
        )
        self.audio_start_time = 0
        self.audio_measured_latency = -1.0
        self.last_dac_time = 0

    def _setup_output_audio(self):
        try:
            self.pa_stream = self.pa.open(
                format=self.pa.get_format_from_width(self.audio.stream.format.bytes),
                channels=self.audio.stream.channels,
                rate=self.audio.stream.rate,
                frames_per_buffer=self.audio.stream.frame_size,
                stream_callback=self.audio_callback,
                output=True,
                start=False,
            )
            logger.debug(
                "Audio output latency: {}".format(self.pa_stream.get_output_latency())
            )
            self.audio_sync = self.pa_stream.get_output_latency()
            self.audio_reported_latency = self.pa_stream.get_output_latency()

        except ValueError:
            self.pa_stream = None

    def _setup_audio_vis(self):
        self.audio_timeline = None
        self.audio_viz_trans = Audio_Viz_Transform(self.g_pool.rec_dir)
        self.audio_viz_data = None
        self.log_scale = False
        self.xlim = (self.g_pool.timestamps[0], self.g_pool.timestamps[-1])
        self.ylim = (0, 210)

    def _setup_filter_graph(self):
        """Graph: buffer -> volume filter -> resample -> buffersink"""
        self.current_audio_volume = self.req_audio_volume
        logger.debug("Setting volume {} ".format(self.current_audio_volume))
        self.filter_graph = av.filter.Graph()
        self.filter_graph_list = []
        self.filter_graph_list.append(
            self.filter_graph.add_buffer(template=self.audio.stream)
        )
        args = "volume={}:precision=float".format(self.current_audio_volume)
        logger.debug("args = {}".format(args))
        self.volume_filter = self.filter_graph.add("volume", args)
        self.filter_graph_list.append(self.volume_filter)
        self.filter_graph_list[-2].link_to(self.filter_graph_list[-1])
        self.filter_graph_list.append(
            self.filter_graph.add(
                "aresample",
                "osf={}".format(self.audio.stream.format.packed.name),
            )
        )
        self.filter_graph_list[-2].link_to(self.filter_graph_list[-1])
        self.filter_graph_list.append(self.filter_graph.add("abuffersink"))
        self.filter_graph_list[-2].link_to(self.filter_graph_list[-1])
        self.filter_graph.configure()

    def sec_to_frames(self, sec):
        return int(np.ceil(sec * self.audio.stream.rate / self.audio.stream.frame_size))

    def frames_to_sec(self, frames):
        return frames * self.audio.stream.frame_size / self.audio.stream.rate

    def buffer_len_secs(self):
        return self.frames_to_sec(len(self.audio_bytes_fifo))

    def audio_callback(self, in_data, frame_count, time_info, status):
        cb_to_adc_time = time_info["output_buffer_dac_time"] - time_info["current_time"]
        start_to_cb_time = monotonic() - self.audio_start_time
        if self.audio_measured_latency < 0:
            self.audio_measured_latency = start_to_cb_time + cb_to_adc_time
            lat_diff = self.audio_reported_latency - self.audio_measured_latency
            self.audio_sync -= lat_diff
            self.g_pool.seek_control.time_slew = self.audio_sync

            logger.debug("Measured latency = {}".format(self.audio_measured_latency))
        self.last_dac_time = time_info["output_buffer_dac_time"]
        if not self.play:
            self.audio_paused = True
            logger.debug("audio cb complete 1")
            return (None, pa.paAbort)
        try:
            samples, ts = self.audio_bytes_fifo.popleft()
            desync = abs(
                self.g_pool.seek_control.current_playback_time + cb_to_adc_time - ts
            )
            if desync > 0.4:
                logger.debug("*** Audio desync detected: {}".format(desync))
                self.audio_paused = True
                return (None, pa.paAbort)
            return (samples, pa.paContinue)
        except IndexError:
            self.audio_paused = True
            logger.debug("audio cb abort 2")
            return (None, pa.paAbort)

    def get_audio_sync(self):
        # Audio has been started without delay
        if self.audio_measured_latency > 0:
            lat_diff = self.pa_stream.get_output_latency() - self.audio_measured_latency
            return self.audio_sync - lat_diff
        else:
            return self.audio_sync

    def get_audio_frame_iterator(self):
        for packet in self.audio.container.demux(self.audio.stream):
            for frame in packet.decode():
                if frame:
                    yield frame

    def audio_idx_to_pts(self, idx):
        return idx * self.audio_pts_rate

    def seek_to_audio_frame(self, seek_pos):
        try:
            self.audio.stream.seek(
                self.audio_start_pts + self.audio_idx_to_pts(seek_pos)
            )
        except av.AVError:
            raise FileSeekError()
        else:
            self.audio_frame_iterator = self.get_audio_frame_iterator()
            self.audio_bytes_fifo.clear()

    def seek_to_frame(self, frame_idx):
        if self.audio.stream is not None:
            audio_idx = bisect(self.audio.timestamps, self.timestamps[frame_idx])
            self.seek_to_audio_frame(audio_idx)

    def on_notify(self, notification):
        if notification["subject"] == "seek_control.was_seeking":
            if self.pa_stream is not None and not self.pa_stream.is_stopped():
                self.pa_stream.stop_stream()
                self.play = False

    def recent_events(self, events):
        self.update_audio_viz()
        self.setup_pyaudio_output_if_necessary()
        if self.pa_stream is None:
            self.play = False
            return
        if self.g_pool.seek_control.playback_speed != 1.0:
            if not self.pa_stream.is_stopped():
                self.pa_stream.stop_stream()
            self.play = False
            return

        self.check_audio_part_setup()
        start_stream = False
        self.play = True
        is_stream_paused = self.pa_stream.is_stopped() or self.audio_paused
        is_audio_delay_low_enough = self.audio_delay <= 0.001
        if is_stream_paused and is_audio_delay_low_enough:
            ts_audio_start = self.calc_audio_start_ts()
            if ts_audio_start is not None:
                start_stream = True

        self.adjust_audio_volume_filter_if_necessary()
        self.fill_audio_queue()

        if start_stream:
            self.calculate_delays(ts_audio_start)
            self.start_audio()

    def calc_audio_start_ts(self):
        pbt = self.g_pool.seek_control.current_playback_time
        audio_start, audio_end = self.audio.timestamps[[0, -1]]

        if audio_start <= pbt <= audio_end:
            audio_idx = bisect(self.audio.timestamps, pbt)
            self.seek_to_audio_frame(audio_idx)
            return self.audio.timestamps[audio_idx]

    def update_audio_viz(self):
        if self.audio_viz_trans is not None:
            self.audio_viz_data, finished = self.audio_viz_trans.get_data(
                log_scale=self.log_scale
            )
            if not finished:
                self.audio_timeline.refresh()

    def setup_pyaudio_output_if_necessary(self):
        if self.pa_stream is not None and not self.pa_stream.is_stopped():
            if not self.audio_paused and not self.pa_stream.is_active():
                logger.debug("Reopening audio stream...")
                self._setup_output_audio()

    def calculate_delays(self, ts_audio_start):
        real_time_delay = (
            ts_audio_start - self.g_pool.seek_control.current_playback_time
        )
        adjusted_delay = real_time_delay - self.pa_stream.get_output_latency()
        self.audio_delay = 0
        self.audio_sync = 0
        if adjusted_delay > 0:
            self.audio_delay = adjusted_delay
            self.audio_sync = 0
        else:
            self.audio_sync = -adjusted_delay

        logger.debug(
            "Audio sync = {} rt_delay = {} adj_delay = {}".format(
                self.audio_sync, real_time_delay, adjusted_delay
            )
        )
        self.g_pool.seek_control.time_slew = self.audio_sync
        self.pa_stream.stop_stream()
        self.audio_measured_latency = -1

    def start_audio(self):
        if self.audio_delay < 0.001:
            self.audio_start_time = monotonic()
            self.pa_stream.start_stream()
        else:

            def delayed_audio_start():
                if self.pa_stream.is_stopped():
                    self.audio_start_time = monotonic()
                    self.pa_stream.start_stream()
                    self.audio_delay = 0
                    logger.debug("Started delayed audio")
                self.audio_timer.cancel()
                self.audio_timer = None

            logger.debug("Starting delayed audio timer")
            self.audio_timer = Timer(self.audio_delay, delayed_audio_start)
            self.audio_timer.start()

        self.audio_paused = False

    def adjust_audio_volume_filter_if_necessary(self):
        if self.filter_graph_list is None:
            self._setup_filter_graph()
        elif self.req_audio_volume != self.current_audio_volume:
            self.current_audio_volume = self.req_audio_volume
            args = "{}".format(self.current_audio_volume)
            self.volume_filter.cmd("volume", args)

    def fill_audio_queue(self):
        frames_to_fetch = self.sec_to_frames(
            max(0, self.req_buffer_size_secs - self.buffer_len_secs())
        )
        if frames_to_fetch <= 0:
            return
        frames_chunk = itertools.islice(self.audio_frame_iterator, frames_to_fetch)
        for audio_frame_p in frames_chunk:
            pts = audio_frame_p.pts
            audio_frame_p.pts = None
            if self.filter_graph_list is not None:
                self.filter_graph_list[0].push(audio_frame_p)
                audio_frame = self.filter_graph_list[-1].pull()
            else:
                audio_frame = self.audio_resampler.resample(audio_frame_p)
            audio_frame.pts = pts
            audio_buffer = bytes(audio_frame.planes[0])

            audio_part_start_ts = self.audio.timestamps[0]
            audio_part_progress = audio_frame.pts * self.audio.stream.time_base
            audio_playback_time = audio_part_start_ts + audio_part_progress
            self.audio_bytes_fifo.append((audio_buffer, audio_playback_time))

    def draw_audio(self, width, height, scale):
        if self.audio_viz_data is None:
            return
        with gl_utils.Coord_System(*self.xlim, *self.ylim):
            pyglui_utils.draw_bars_buffer(self.audio_viz_data, color=viz_color)

    def init_ui(self):
        if self.pa_stream is None:
            return
        self.add_menu()
        self.menu_icon.order = 0.01
        self.menu.label = "Audio Playback"

        def set_volume(val):
            self.req_audio_volume = val

        self.menu.append(
            ui.Slider(
                "req_audio_volume",
                self,
                step=0.05,
                min=0.0,
                max=1.0,
                label="Volume",
            )
        )
        self.menu.append(
            ui.Slider(
                "req_buffer_size_secs",
                self,
                step=0.05,
                min=0.0,
                max=1.0,
                label="Buffer size (s)",
            )
        )

        self.audio_timeline = ui.Timeline("Audio level", self.draw_audio, None)
        self.audio_timeline.content_height *= 2
        self.g_pool.user_timelines.append(self.audio_timeline)
        self.menu.append(ui.Switch("log_scale", self, label="Log scale"))

    def cleanup(self):
        if self.audio_timer is not None:
            self.audio_timer.cancel()
            self.audio_timer = None

    def check_ts_consistency(self, reference_frame):
        if self.should_check_ts_consistency:
            print("**** Checking stream")
            for i, af in enumerate(self.audio_frame_iterator):
                fnum = i + 1
                if af.samples != reference_frame.samples:
                    print("fnum {} samples = {}".format(fnum, af.samples))
                if af.pts != self.audio_idx_to_pts(fnum):
                    print(
                        "af.pts = {} fnum = {} idx2pts = {}".format(
                            af.pts, fnum, self.audio_idx_to_pts(fnum)
                        )
                    )
                if (
                    self.audio.timestamps[fnum]
                    != self.audio.timestamps[0] + af.pts * self.audio.stream.time_base
                ):
                    print(
                        "ts[0] + af.pts = {} fnum = {} timestamp = {}".format(
                            self.audio.timestamps[0]
                            + af.pts * self.audio.stream.time_base,
                            fnum,
                            self.audio.timestamps[fnum],
                        )
                    )
            print("**** Done")
