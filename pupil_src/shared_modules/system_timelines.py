'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
import numpy as np

from pyglui import ui
from pyglui.pyfontstash import fontstash as fs
from pyglui.cygl.utils import *
import OpenGL.GL as gl

from plugin import System_Plugin_Base
import gl_utils
import av
# logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logger.DEBUG)

#av.logging.set_level(av.logging.DEBUG)
#logging.getLogger('libav').setLevel(logging.DEBUG)

world_color = RGBA(0.66, 0.86, 0.461, 1.)
right_color = RGBA(0.9844, 0.5938, 0.4023, 1.)
left_color = RGBA(0.668, 0.6133, 0.9453, 1.)

class Audio_Viz_Transform():
    def __init__(self, rec_dir, sps_rate=60):
        import av
        import os
        import errno
        audio_file = os.path.join(rec_dir, 'audio.mp4')
        if os.path.isfile(audio_file):
            self.audio_container = av.open(str(audio_file))
            try:
                self.audio_stream = next(s for s in self.audio_container.streams if s.type == 'audio')
                logger.debug("loaded audiostream: %s" % self.audio_stream)
            except StopIteration:
                audio_stream = None
                logger.debug("No audiostream found in media container")
                return
        else:
            raise FileNotFoundError(errno.ENOENT, audio_file)
        if self.audio_stream  is not None:
            audiots_path = os.path.splitext(audio_file)[0] + '_timestamps.npy'
            try:
                self.audio_timestamps = np.load(audiots_path)
            except IOError:
                self.audio_timestamps = None
                logger.warning("Could not load audio timestamps")
                raise FileNotFoundError(errno.ENOENT, audiots_path)
        self.sps_rate = sps_rate
        self.start_ts = self.audio_timestamps[0]

        # Test lowpass filtering + viz
        self.lp_graph = av.filter.Graph()
        self.lp_graph_list = []
        self.lp_graph_list.append(self.lp_graph.add_buffer(template=self.audio_stream))
        args = "f=10"
        print("args = {}".format(args))
        # lp_graph_list.append(lp_graph.add("lowpass", args))
        # "attacks=.1|.1:decays=.2|.2:points=.-900/-900|-50.1/-900|-50/-50:soft-knee=.01:gain=0:volume=-90:delay=.1")
        self.lp_graph_list.append(self.lp_graph.add("compand", ".1|.1:.2|.2:-900/-900|-50.1/-900|-50/-50:.01:0:-90:.1"))
        self.lp_graph_list[-2].link_to(self.lp_graph_list[-1])
        # lp_graph_list.append(lp_graph.add("aresample", "osr=30"))
        # lp_graph_list[-2].link_to(lp_graph_list[-1])
        self.lp_graph_list.append(self.lp_graph.add("abuffersink"))
        self.lp_graph_list[-2].link_to(self.lp_graph_list[-1])
        self.lp_graph.configure()

        #audio_resampler1 = av.audio.resampler.AudioResampler(format=av.AudioFormat('dblp'),
        #                                                     layout=audio_stream.layout,
        #                                                     rate=audio_stream.rate)
        self.audio_resampler = av.audio.resampler.AudioResampler(format=self.audio_stream.format,
                                                            layout=self.audio_stream.layout,
                                                            rate=60)
        self.next_audio_frame = self._next_audio_frame()
        self.samples_min = 999.
        self.samples_max = -999.

    def _next_audio_frame(self):
        for packet in self.audio_container.demux(self.audio_stream):
            for frame in packet.decode():
                if frame:
                    yield frame
        raise StopIteration()

    def sec_to_frames(self, sec):
        return int(np.ceil(sec * self.audio_stream.rate / self.audio_stream.frame_size))

    def get_data(self, seconds=30.):
        import itertools
        allSamples = None
        frames_to_fetch = self.sec_to_frames(seconds)
        if frames_to_fetch > 0:
            frames_chunk = itertools.islice(self.next_audio_frame, frames_to_fetch)
        for af in frames_chunk:
            audio_frame = af
            # af_dbl = audio_resampler1.resample(af)
            # lp_graph_list[0].push(af)
            # audio_frame = lp_graph_list[-1].pull()
            # if audio_frame is None:
            #    continue
            # audio_frame.pts = None
            audio_frame_rs = self.audio_resampler.resample(audio_frame)
            if audio_frame_rs is None:
                continue
            samples = np.frombuffer(audio_frame_rs.planes[0], dtype=np.float32)
            if allSamples is not None:
                allSamples = np.concatenate((allSamples, samples), axis=0)
            else:
                allSamples = samples
        if allSamples is None:
            return
        # flush
        audio_frame_rs = self.audio_resampler.resample(None)
        if audio_frame_rs is not None:
            samples = np.frombuffer(audio_frame_rs.planes[0], dtype=np.float32)
            if allSamples is not None:
                allSamples = np.concatenate((allSamples, samples), axis=0)
            else:
                allSamples = samples

        new_ts = np.arange(0, len(allSamples), 1) / self.audio_resampler.rate
        new_ts += self.start_ts
        self.start_ts = new_ts[-1] + 1 / self.audio_resampler.rate

        abs_samples = np.abs(allSamples)

        # TODO: handle min/max adequately

        scaled_samples = (abs_samples - abs_samples.min()) / (abs_samples.max() - abs_samples.min())
        a_levels = [alevel for alevel in zip(new_ts, scaled_samples)]

        return a_levels




class System_Timelines(System_Plugin_Base):
    def __init__(self, g_pool, show_world_fps=True, show_eye_fps=True):
        super().__init__(g_pool)
        self.show_world_fps = show_world_fps
        self.show_eye_fps = show_eye_fps
        self.cache = {}
        self.aud_viz_trans = None
        self.get_audio_data = True
        self.cache_fps_data()
        self.cache_audio_data()

    def init_ui(self):
        self.glfont = fs.Context()
        self.glfont.add_font('opensans', ui.get_opensans_font_path())
        self.glfont.set_font('opensans')
        self.fps_timeline = ui.Timeline('Recorded FPS', self.draw_fps, self.draw_fps_legend)
        self.fps_timeline.content_height *= 2
        self.g_pool.user_timelines.append(self.fps_timeline)
        self.audio_timeline = ui.Timeline('Audio level', self.draw_audio, None)
        self.audio_timeline.content_height *= 2
        self.g_pool.user_timelines.append(self.audio_timeline)

    def deinit_ui(self):
        self.g_pool.user_timelines.remove(self.fps_timeline)
        self.fps_timeline = None

    def cache_fps_data(self):
        t0, t1 = self.g_pool.timestamps[0], self.g_pool.timestamps[-1]

        w_ts = np.asarray(self.g_pool.timestamps)
        w_fps = 1. / np.diff(w_ts)
        w_fps = [fps for fps in zip(w_ts, w_fps)]

        e0_ts = np.array([p['timestamp'] for p in self.g_pool.pupil_positions if p['id'] == 0])
        if e0_ts.shape[0] > 1:
            e0_fps = 1. / np.diff(e0_ts)
            e0_fps = [fps for fps in zip(e0_ts, e0_fps)]
        else:
            e0_fps = []

        e1_ts = np.array([p['timestamp'] for p in self.g_pool.pupil_positions if p['id'] == 1])
        if e1_ts.shape[0] > 1:
            e1_fps = 1. / np.diff(e1_ts)
            e1_fps = [fps for fps in zip(e1_ts, e1_fps)]
        else:
            e1_fps = []

        #self.cache = {'world': w_fps, 'eye0': e0_fps, 'eye1': e1_fps,
        #              'xlim': [t0, t1], 'ylim': [0, 210]}
        self.cache['world'] = w_fps
        self.cache['eye0'] = e0_fps
        self.cache['eye1'] = e1_fps
        self.cache['xlim'] = [t0, t1]
        self.cache['ylim'] = [0, 210]



    def cache_audio_data(self):

        #import av
        #import os
        #audio_file = os.path.join(self.g_pool.rec_dir, 'audio.mp4')
        #if os.path.isfile(audio_file):
        #    audio_container = av.open(str(audio_file))
        #    try:
        #        audio_stream = next(s for s in audio_container.streams if s.type == 'audio')
        #        logger.debug("loaded audiostream: %s" % audio_stream)
        #    except StopIteration:
        #        audio_stream = None
        #        logger.debug("No audiostream found in media container")
        #        return
        #else:
        #    return
        #if audio_stream is not None:
        #    audiots_path = os.path.splitext(audio_file)[0] + '_timestamps.npy'
        #    try:
        #        audio_timestamps = np.load(audiots_path)
        #    except IOError:
        #        audio_timestamps = None
        #        logger.warning("Could not load audio timestamps")
        #        return

        #def next_audio_frame():
        #    for packet in audio_container.demux(audio_stream):
        #        for frame in packet.decode():
        #            if frame:
        #                yield frame
        #    raise StopIteration()
        ## Test lowpass filtering + viz
        #lp_graph = av.filter.Graph()
        #lp_graph_list = []
        #lp_graph_list.append(lp_graph.add_buffer(template=audio_stream))
        #args = "f=10"
        #print("args = {}".format(args))
        ##lp_graph_list.append(lp_graph.add("lowpass", args))
        ## "attacks=.1|.1:decays=.2|.2:points=.-900/-900|-50.1/-900|-50/-50:soft-knee=.01:gain=0:volume=-90:delay=.1")
        #lp_graph_list.append(lp_graph.add("compand", ".1|.1:.2|.2:-900/-900|-50.1/-900|-50/-50:.01:0:-90:.1"))
        #lp_graph_list[-2].link_to(lp_graph_list[-1])
        ## lp_graph_list.append(lp_graph.add("aresample", "osr=30"))
        ## lp_graph_list[-2].link_to(lp_graph_list[-1])
        #lp_graph_list.append(lp_graph.add("abuffersink"))
        #lp_graph_list[-2].link_to(lp_graph_list[-1])
        #lp_graph.configure()

        #audio_resampler1 = av.audio.resampler.AudioResampler(format=av.AudioFormat('dblp'),
        #                                                    layout=audio_stream.layout,
        #                                                    rate=audio_stream.rate)
        #audio_resampler = av.audio.resampler.AudioResampler(format=audio_stream.format,
        #                                                    layout=audio_stream.layout,
        #                                                    rate=60)
        ##frames_chunk = itertools.islice(next_audio_frame, int(5 * 48000 / 1024))
        #allSamples = None
        #for af in next_audio_frame():
        #    audio_frame = af
        #    #af_dbl = audio_resampler1.resample(af)
        #    #lp_graph_list[0].push(af)
        #    #audio_frame = lp_graph_list[-1].pull()
        #    #if audio_frame is None:
        #    #    continue
        #    #audio_frame.pts = None
        #    audio_frame_rs = audio_resampler.resample(audio_frame)
        #    if audio_frame_rs is None:
        #        continue
        #    samples = np.frombuffer(audio_frame_rs.planes[0], dtype=np.float32)
        #    if allSamples is not None:
        #        allSamples = np.concatenate((allSamples, samples), axis=0)
        #    else:
        #        allSamples = samples
        ##flush
        #audio_frame_rs = audio_resampler.resample(None)
        #if audio_frame_rs is not None:
        #    samples = np.frombuffer(audio_frame_rs.planes[0], dtype=np.float32)
        #    if allSamples is not None:
        #        allSamples = np.concatenate((allSamples, samples), axis=0)
        #    else:
        #        allSamples = samples


        #new_ts = np.arange(0, len(allSamples),  1) / audio_resampler.rate
        #new_ts += audio_timestamps[0]

        #scaled_samples = np.abs((allSamples - allSamples.min()) / (allSamples.max() - allSamples.min()))
        #a_levels = [alevel for alevel in zip(new_ts, scaled_samples)]
        if self.get_audio_data:
            if self.aud_viz_trans is None:
                try:
                    self.aud_viz_trans = Audio_Viz_Transform(self.g_pool.rec_dir)
                except FileNotFoundError:
                    self.get_audio_data = False
                    return

            a_levels = self.aud_viz_trans.get_data()
            if a_levels is not None:
                if 'audio_level' not in self.cache.keys():
                    self.cache['audio_level'] = a_levels
                else:
                    self.cache['audio_level'] = np.concatenate((self.cache['audio_level'] , a_levels), axis=0)
            else:
                self.get_audio_data = False
        return self.get_audio_data
        # shift_samples = scaled_samples - scaled_samples/2
        # plt.bar(index, np.abs(scaled_samples))
        # plt.show()

    def draw_audio(self, width, height, scale):
        with gl_utils.Coord_System(*self.cache['xlim'], *self.cache['ylim']):
            draw_bars2(self.cache['audio_level'], 210, color=right_color)


    def draw_fps(self, width, height, scale):
        with gl_utils.Coord_System(*self.cache['xlim'], *self.cache['ylim']):
            if self.show_world_fps:
                draw_points(self.cache['world'], size=2*scale, color=world_color)
            if self.show_eye_fps:
                draw_points(self.cache['eye0'], size=2*scale, color=right_color)
                draw_points(self.cache['eye1'], size=2*scale, color=left_color)

    def draw_fps_legend(self, width, height, scale):
        self.glfont.push_state()
        self.glfont.set_align_string(v_align='right', h_align='top')
        self.glfont.set_size(15. * scale)
        self.glfont.draw_text(width, 0, self.fps_timeline.label)

        legend_height = 13. * scale
        pad = 10 * scale

        if self.show_world_fps:
            self.glfont.draw_text(width, legend_height, 'world FPS')
            draw_polyline([(pad, legend_height + pad * 2 / 3),
                           (width / 2, legend_height + pad * 2 / 3)],
                          color=world_color, line_type=gl.GL_LINES, thickness=4.*scale)
            legend_height += 1.5 * pad

        if self.show_eye_fps:
            self.glfont.draw_text(width, legend_height, 'eye1 FPS')
            draw_polyline([(pad, legend_height + pad * 2 / 3),
                           (width / 2, legend_height + pad * 2 / 3)],
                          color=left_color, line_type=gl.GL_LINES, thickness=4.*scale)
            legend_height += 1.5 * pad

            self.glfont.draw_text(width, legend_height, 'eye0 FPS')
            draw_polyline([(pad, legend_height + pad * 2 / 3),
                           (width / 2, legend_height + pad * 2 / 3)],
                          color=right_color, line_type=gl.GL_LINES, thickness=4.*scale)

        self.glfont.pop_state()

    def on_notify(self, notification):
        if notification['subject'] == 'pupil_positions_changed':
            self.cache_fps_data()
            self.fps_timeline.refresh()

    def recent_events(self, events):
        if self.cache_audio_data():
            self.audio_timeline.refresh()
