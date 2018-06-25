'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os
import av
from time import sleep

from .base_backend import Base_Source, Playback_Source, Base_Manager, EndofVideoError
from camera_models import load_intrinsics

import numpy as np
from multiprocessing import cpu_count
import os.path
from fractions import Fraction

# logging
import logging
logger = logging.getLogger(__name__)

assert av.__version__ >= '0.2.5'
av.logging.set_level(av.logging.ERROR)
logging.getLogger('libav').setLevel(logging.ERROR)


class FileSeekError(Exception):
    pass


class Frame(object):
    """docstring of Frame"""
    def __init__(self, timestamp, av_frame, index):
        self._av_frame = av_frame
        self.timestamp = timestamp
        self.index = index
        self._img = None
        self._gray = None
        self.jpeg_buffer = None
        self.yuv_buffer = None
        self.height, self.width = av_frame.height, av_frame.width

    def copy(self):
        return Frame(self.timestamp, self._av_frame, self.index)

    @property
    def img(self):
        if self._img is None:
            self._img = self._av_frame.to_nd_array(format='bgr24')
        return self._img

    @property
    def bgr(self):
        return self.img

    @property
    def gray(self):
        if self._gray is None:
            plane = self._av_frame.planes[0]
            self._gray = np.frombuffer(plane, np.uint8)
            try:
                self._gray.shape = self.height, self.width
            except ValueError:
                self._gray = self._gray.reshape(-1, plane.line_size)
                self._gray = np.ascontiguousarray(self._gray[:, :self.width])
        return self._gray


class File_Source(Playback_Source, Base_Source):
    """Simple file capture.

    Attributes:
        source_path (str): Path to source file
        timestamps (str): Path to timestamps file
    """

    def __init__(self, g_pool, source_path=None, loop=False, *args, **kwargs):
        super().__init__(g_pool, *args, **kwargs)
        if self.timing == 'external':
            self.recent_events = self.recent_events_external_timing
        else:
            self.recent_events = self.recent_events_own_timing

        # minimal attribute set
        self._initialised = True
        self.source_path = source_path
        self.timestamps = None
        self.loop = loop

        if not source_path or not os.path.isfile(source_path):
            logger.error('Init failed. Source file could not be found at `%s`'%source_path)
            self._initialised = False
            return

        self.container = av.open(str(source_path))

        try:
            self.video_stream = next(s for s in self.container.streams if s.type=="video")# looking for the first videostream
            logger.debug("loaded videostream: %s"%self.video_stream)
            self.video_stream.thread_count = cpu_count()
        except StopIteration:
            self.video_stream = None
            logger.error("No videostream found in media container")

        try:
            self.audio_stream = next(s for s in self.container.streams if s.type=='audio')# looking for the first audiostream
            logger.debug("loaded audiostream: %s"%self.audio_stream)
        except StopIteration:
            self.audio_stream = None
            logger.debug("No audiostream found in media container")

        if not self.video_stream and not self.audio_stream:
            logger.error('Init failed. Could not find any video or audio stream in the given source file.')
            self._initialised = False
            return

        self.target_frame_idx = 0
        self.current_frame_idx = 0

        # we will use below for av playback
        # self.selected_streams = [s for s in (self.video_stream,self.audio_stream) if s]
        # self.av_packet_iterator = self.container.demux(self.selected_streams)

        avg_rate = self.video_stream.average_rate
        if avg_rate is None:
            avg_rate = Fraction(0,1)

        if float(avg_rate)%1 != 0.0:
            logger.error('Videofile pts are not evenly spaced, pts to index conversion may fail and be inconsitent.')

        # load/generate timestamps.
        timestamps_path, ext = os.path.splitext(source_path)
        timestamps_path += '_timestamps.npy'
        try:
            self.timestamps = np.load(timestamps_path)
        except IOError:
            logger.warning("did not find timestamps file, making timetamps up based on fps and frame count. Frame count and timestamps are not accurate!")
            frame_rate = float(avg_rate)
            self.timestamps = [i/frame_rate for i in range(int(self.container.duration/av.time_base*frame_rate)+100)]  # we are adding some slack.
        else:
            logger.debug("Auto loaded %s timestamps from %s" % (len(self.timestamps), timestamps_path))
        assert isinstance(self.timestamps[0], float), 'Timestamps need to be instances of python float, got {}'.format(type(self.timestamps[0]))
        self.timestamps = self.timestamps

        # set the pts rate to convert pts to frame index. We use videos with pts writte like indecies.
        self.next_frame = self._next_frame()
        f0, f1 = next(self.next_frame), next(self.next_frame)
        self.pts_rate = f1.pts
        self.seek_to_frame(0)
        self.average_rate = (self.timestamps[-1]-self.timestamps[0])/len(self.timestamps)

        loc, name = os.path.split(os.path.splitext(source_path)[0])
        self._intrinsics = load_intrinsics(loc, name, self.frame_size)

    def ensure_initialisation(fallback_func=None, requires_playback=False):
        from functools import wraps

        def decorator(func):
            @wraps(func)
            def run_func(self, *args, **kwargs):
                if self._initialised and self.video_stream:
                    # test self.play only if requires_playback is True
                    if not requires_playback or self.play:
                        return func(self, *args, **kwargs)
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                else:
                    logger.debug('Initialisation required.')
            return run_func
        return decorator

    @property
    def initialised(self):
        return self._initialised

    @property
    @ensure_initialisation(fallback_func=lambda: (640, 480))
    def frame_size(self):
        return int(self.video_stream.format.width), int(self.video_stream.format.height)

    @property
    @ensure_initialisation(fallback_func=lambda: 20)
    def frame_rate(self):
        return 1./float(self.average_rate)

    def get_init_dict(self):
        if self.g_pool.app == 'capture':
            settings = super().get_init_dict()
            settings['source_path'] = self.source_path
            settings['loop'] = self.loop
            return settings
        else:
            raise NotImplementedError()

    @property
    def name(self):
        if self.source_path:
            return os.path.splitext(self.source_path)[0]
        else:
            return 'File source in ghost mode'

    def get_frame_index(self):
        return self.current_frame_idx

    def get_frame_count(self):
        return len(self.timestamps)

    @ensure_initialisation()
    def _next_frame(self):
        for packet in self.container.demux(self.video_stream):
            for frame in packet.decode():
                if frame:
                    yield frame
        raise StopIteration()

    @ensure_initialisation()
    def pts_to_idx(self, pts):
        # some older mkv did not use perfect timestamping so we are doing int(round()) to clear that.
        # With properly spaced pts (any v0.6.100+ recording) just int() would suffice.
        # print float(pts*self.video_stream.time_base*self.video_stream.average_rate),round(pts*self.video_stream.time_base*self.video_stream.average_rate)
        return int(pts/self.pts_rate)

    @ensure_initialisation()
    def idx_to_pts(self, idx):
        return idx*self.pts_rate

    @ensure_initialisation()
    def get_frame(self):
        frame = None
        for frame in self.next_frame:
            index = self.pts_to_idx(frame.pts)
            if index == self.target_frame_idx:
                break
            elif index < self.target_frame_idx:
                pass
                # logger.info('Frame index not consistent. Skipping forward')
            else:
                logger.debug('Frame index not consistent.')
                break
        if not frame:
            if self.loop:
                logger.info('Looping enabled. Seeking to beginning.')
                self.seek_to_frame(0)
                self.target_frame_idx = 0
                return self.get_frame()
            else:
                logger.debug("End of videofile %s %s"%(self.current_frame_idx,len(self.timestamps)))
                raise EndofVideoError('Reached end of video file')
        try:
            timestamp = self.timestamps[index]
        except IndexError:
            logger.info("Reached end of timestamps list.")
            raise EndofVideoError("Reached end of timestamps list.")

        self.target_frame_idx = index+1
        self.current_frame_idx = index
        return Frame(timestamp, frame, index=index)

    @ensure_initialisation(fallback_func=lambda evt: sleep(0.05))
    def recent_events_external_timing(self, events):
        try:
            last_index = self._recent_frame.index
        except AttributeError:
            # called once on start when self._recent_frame is None
            last_index = -1

        frame = None
        pbt = self.g_pool.seek_control.current_playback_time
        ts_idx = self.g_pool.seek_control.ts_idx_from_playback_time(pbt)
        if ts_idx == last_index:
            frame = self._recent_frame.copy()
            if self.play and ts_idx == self.get_frame_count() - 1:
                logger.info('Video has ended.')
                self.g_pool.seek_control.play = False

        elif ts_idx < last_index or ts_idx > last_index + 1:
            # time to seek
            self.seek_to_frame(ts_idx)

        # Only call get_frame() if the next frame is actually needed
        try:
            frame = frame or self.get_frame()
        except EndofVideoError:
            logger.info('Video has ended.')
            self.g_pool.seek_control.play = False
            frame = frame or self._recent_frame.copy()

        events['frame'] = frame
        self._recent_frame = frame

    @ensure_initialisation(fallback_func=lambda evt: sleep(0.05),
                           requires_playback=True)
    def recent_events_own_timing(self, events):
        try:
            frame = self.get_frame()
        except EndofVideoError:
            logger.info('Video has ended.')
            self.notify_all({"subject": 'file_source.video_finished', 'source_path': self.source_path})
            self.play = False
        else:
            if self.timing:
                self.wait(frame.timestamp)
            self._recent_frame = frame
            events['frame'] = frame

    @ensure_initialisation()
    def seek_to_frame(self, seek_pos):
        # frame accurate seeking
        try:
            self.video_stream.seek(int(self.idx_to_pts(seek_pos)))
        except av.AVError as e:
            raise FileSeekError()
        else:
            self.next_frame = self._next_frame()
            self.finished_sleep = 0
            self.target_frame_idx = seek_pos

    def on_notify(self, notification):
        if notification['subject'] == 'file_source.seek' and notification.get('source_path') == self.source_path:
            self.seek_to_frame(notification['frame_index'])
        elif notification['subject'] == 'file_source.should_play' and notification.get('source_path') == self.source_path:
            self.play = True
        elif notification['subject'] == 'file_source.should_pause' and notification.get('source_path') == self.source_path:
            self.play = False

    def seek_to_prev_frame(self):
        self.seek_to_frame(max(0, self.current_frame_idx - 1))

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'File Source: {}'.format(os.path.split(self.source_path)[-1])
        from pyglui import ui

        self.menu.append(ui.Info_Text('The file source plugin loads and displays video from a given file.'))

        if self.g_pool.app == 'capture':
            def toggle_looping(val):
                self.loop = val
                if val:
                    self.play = True

            self.menu.append(ui.Switch('loop', self, setter=toggle_looping))

        self.menu.append(ui.Text_Input('source_path', self, label='Full path',
                                       setter=lambda x: None))

        self.menu.append(ui.Text_Input('frame_size', label='Frame size',
                                       setter=lambda x: None,
                                       getter=lambda: '{} x {}'.format(*self.frame_size)))

        self.menu.append(ui.Text_Input('frame_rate', label='Frame rate',
                                       setter=lambda x: None,
                                       getter=lambda: '{:.0f} FPS'.format(self.frame_rate)))

        self.menu.append(ui.Text_Input('frame_num', label='Number of frames',
                                       setter=lambda x: None,
                                       getter=lambda: self.get_frame_count()))
    def deinit_ui(self):
        self.remove_menu()

    @property
    def jpeg_support(self):
        return False


class File_Manager(Base_Manager):
    """Summary

    Attributes:
        file_exts (list): File extensions to filter displayed files
        root_folder (str): Folder path, which includes file sources
    """
    gui_name = 'Video File Source'
    file_exts = ['.mp4', '.mkv', '.mov', '.mjpeg']

    def __init__(self, g_pool, root_folder=None):
        super().__init__(g_pool)
        base_dir = self.g_pool.user_dir.rsplit(os.path.sep,1)[0]
        default_rec_dir = os.path.join(base_dir,'recordings')
        self.root_folder = root_folder or default_rec_dir

    def init_ui(self):
        self.add_menu()
        from pyglui import ui
        self.menu.append(ui.Info_Text('Enter a folder to enumerate all eligible video files. Be aware that entering folders with a lot of files can slow down Pupil Capture.'))

        def set_root(folder):
            if not os.path.isdir(folder):
                logger.error('`%s` is not a valid folder path.'%folder)
            else: self.root_folder = folder

        self.menu.append(ui.Text_Input('root_folder',self,label='Source Folder',setter=set_root))

        def split_enumeration():
            eligible_files = self.enumerate_folder(self.root_folder)
            eligible_files.insert(0, (None, 'Select to activate'))
            return zip(*eligible_files)

        self.menu.append(ui.Selector(
            'selected_file',
            selection_getter=split_enumeration,
            getter=lambda: None,
            setter=self.activate,
            label='Video File'
        ))

    def deinit_ui(self):
        self.remove_menu()

    def activate(self, full_path):
        if not full_path:
            return
        settings = {
            'source_path': full_path,
            'timing': 'own'
        }
        self.activate_source(settings)

    def on_drop(self, paths):
        for p in paths:
            if os.path.splitext(p)[-1] in self.file_exts:
                self.activate(p)
                return

    def enumerate_folder(self,path):
        eligible_files  = []
        is_eligible = lambda f: os.path.splitext(f)[-1] in self.file_exts
        path = os.path.abspath(os.path.expanduser(path))
        for root,dirs,files in os.walk(path):
            def root_split(file):
                full_p = os.path.join(root,file)
                disp_p = full_p.replace(path,'')
                return (full_p, disp_p)
            eligible_files.extend(map(root_split, filter(is_eligible, files)))
        eligible_files.sort(key=lambda x: x[1])
        return eligible_files

    def get_init_dict(self):
        return {'root_folder':self.root_folder}

    def activate_source(self, settings={}):
        if self.g_pool.process == 'world':
            self.notify_all({'subject':'start_plugin',"name":"File_Source",'args':settings})
        else:
            self.notify_all({'subject':'start_eye_capture','target':self.g_pool.process, "name":"File_Source",'args':settings})

    def recent_events(self,events):
        pass
