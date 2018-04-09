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
import av
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logger.DEBUG)

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
                self.audio_stream = None
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
        #self.lp_graph = av.filter.Graph()
        #self.lp_graph_list = []
        #self.lp_graph_list.append(self.lp_graph.add_buffer(template=self.audio_stream))
        #args = "f=10"
        #print("args = {}".format(args))
        ## lp_graph_list.append(lp_graph.add("lowpass", args))
        ## "attacks=.1|.1:decays=.2|.2:points=.-900/-900|-50.1/-900|-50/-50:soft-knee=.01:gain=0:volume=-90:delay=.1")
        #self.lp_graph_list.append(self.lp_graph.add("compand", ".1|.1:.2|.2:-900/-900|-50.1/-900|-50/-50:.01:0:-90:.1"))
        #self.lp_graph_list[-2].link_to(self.lp_graph_list[-1])
        ## lp_graph_list.append(lp_graph.add("aresample", "osr=30"))
        ## lp_graph_list[-2].link_to(lp_graph_list[-1])
        #self.lp_graph_list.append(self.lp_graph.add("abuffersink"))
        #self.lp_graph_list[-2].link_to(self.lp_graph_list[-1])
        #self.lp_graph.configure()

        #audio_resampler1 = av.audio.resampler.AudioResampler(format=av.AudioFormat('dblp'),
        #                                                     layout=audio_stream.layout,
        #                                                     rate=audio_stream.rate)
        self.audio_resampler = av.audio.resampler.AudioResampler(format=self.audio_stream.format,
                                                            layout=self.audio_stream.layout,
                                                            rate=60)
        self.next_audio_frame = self._next_audio_frame()
        self.all_abs_samples = None
        self.finished = False
        self.a_levels = None
        self.a_levels_log = None
        self.final_rescale = True
        self.log_scaling = False

    def _next_audio_frame(self):
        for packet in self.audio_container.demux(self.audio_stream):
            for frame in packet.decode():
                if frame:
                    yield frame
        raise StopIteration()

    def sec_to_frames(self, sec):
        return int(np.ceil(sec * self.audio_stream.rate / self.audio_stream.frame_size))

    def get_data(self, seconds=30., height=210, log_scale=False):
        import itertools

        finished = self.finished
        if not finished:
            allSamples = None
            frames_to_fetch = self.sec_to_frames(seconds)
            if frames_to_fetch > 0:
                frames_chunk = itertools.islice(self.next_audio_frame, frames_to_fetch)
            for af in frames_chunk:
                audio_frame = af
                audio_frame.pts = None
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

            # flush
            audio_frame_rs = self.audio_resampler.resample(None)
            if audio_frame_rs is not None:
                samples = np.frombuffer(audio_frame_rs.planes[0], dtype=np.float32)
                if allSamples is not None:
                    allSamples = np.concatenate((allSamples, samples), axis=0)
                else:
                    allSamples = samples
            if allSamples is not None:
                new_ts = np.arange(0, len(allSamples), 1, dtype=np.float32) / self.audio_resampler.rate
                new_ts += self.start_ts
                self.start_ts = new_ts[-1] + 1 / self.audio_resampler.rate

                abs_samples = np.abs(allSamples)
                if self.all_abs_samples is not None:
                    self.all_abs_samples = np.concatenate((self.all_abs_samples, abs_samples), axis=0)
                else:
                    self.all_abs_samples = abs_samples

                scaled_samples_log = self.log_scale(abs_samples)

                if abs_samples.max() - abs_samples.min() > 0.:
                    scaled_samples = (abs_samples - abs_samples.min()) / (abs_samples.max() - abs_samples.min())
                elif abs_samples.max() > 0.:
                    scaled_samples = abs_samples / abs_samples.max()
                else:
                    scaled_samples = abs_samples


            else:
                new_ts = np.arange(0, len(self.all_abs_samples), 1, dtype=np.float32) / self.audio_resampler.rate
                new_ts += self.audio_timestamps[0]

                #self.all_abs_samples = np.log10(self.all_abs_samples)
                self.all_abs_samples[-1] = 0.

                scaled_samples_log = self.log_scale(self.all_abs_samples)

                if self.all_abs_samples.max() - self.all_abs_samples.min() > 0.:
                    scaled_samples = (self.all_abs_samples - self.all_abs_samples.min()) / (self.all_abs_samples.max() - self.all_abs_samples.min())
                elif self.all_abs_samples.max() > 0.:
                    scaled_samples = self.all_abs_samples / self.all_abs_samples.max()
                else:
                    scaled_samples = self.all_abs_samples

                self.a_levels = None
                self.finished = True
            if not self.finished or self.final_rescale:
                a_levels = self.get_verteces(new_ts, scaled_samples, height)

                if self.a_levels is not None:
                    self.a_levels = np.concatenate((self.a_levels, a_levels), axis=0)
                else:
                    self.a_levels = a_levels

                a_levels_log = self.get_verteces(new_ts, scaled_samples_log, height)

                if self.a_levels_log is not None:
                    self.a_levels_log = np.concatenate((self.a_levels_log, a_levels_log), axis=0)
                else:
                    self.a_levels_log = a_levels_log

        if not log_scale:
            ret = self.a_levels
        else:
            ret = self.a_levels_log

        if self.log_scaling != log_scale:
            self.log_scaling = log_scale
            finished = False

        return ret, finished

    def get_verteces(self, new_ts, scaled_samples, height):
        points_y1 = scaled_samples * (-height / 2) + height / 2
        points_xy1 = np.concatenate((new_ts.reshape(-1, 1), points_y1.reshape(-1, 1)), 1).reshape(-1)
        points_y2 = scaled_samples * (height / 2) + height / 2
        points_xy2 = np.concatenate((new_ts.reshape(-1, 1), points_y2.reshape(-1, 1)), 1).reshape(-1)
        # a_levels = [alevel for alevel in zip(new_ts, scaled_samples)]
        a_levels = np.concatenate((points_xy1.reshape(-1, 2), points_xy2.reshape(-1, 2)), 1).reshape(-1)

        return a_levels

    def log_scale(self, abs_samples):
        scaled_samples = abs_samples / abs_samples.max() + .0001
        scaled_samples_log = 10 * np.log10(scaled_samples)
        sc_min = scaled_samples_log.min()
        scaled_samples_log += - sc_min
        sc_max = scaled_samples_log.max()
        scaled_samples_log /= sc_max

        return scaled_samples_log
