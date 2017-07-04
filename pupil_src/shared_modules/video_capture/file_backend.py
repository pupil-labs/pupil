'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os,sys
import av
assert av.__version__ >= '0.2.5'


from .base_backend import Base_Source, Base_Manager

import numpy as np
from time import time,sleep
from fractions import Fraction
from  multiprocessing import cpu_count
import os.path

#logging
import logging
logger = logging.getLogger(__name__)

av.logging.set_level(av.logging.ERROR)
logging.getLogger('libav').setLevel(logging.ERROR)

class FileCaptureError(Exception):
    """General Exception for this module"""
    def __init__(self, arg):
        super().__init__()
        self.arg = arg


class EndofVideoFileError(Exception):
    """docstring for EndofVideoFileError"""
    def __init__(self, arg):
        super().__init__()
        self.arg = arg


class FileSeekError(Exception):
    """docstring for EndofVideoFileError"""
    def __init__(self):
        super().__init__()


class Frame(object):
    """docstring of Frame"""
    def __init__(self, timestamp,av_frame,index):
        self._av_frame = av_frame
        self.timestamp = timestamp
        self.index = index
        self._img = None
        self._gray = None
        self.jpeg_buffer = None
        self.yuv_buffer = None
        self.height,self.width = av_frame.height,av_frame.width

    def copy(self):
        return Frame(self.timestamp,self._av_frame,self.index)

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
            self._gray = np.frombuffer(self._av_frame.planes[0], np.uint8).reshape(self.height,self.width)
        return self._gray


class File_Source(Base_Source):
    """Simple file capture.

    Attributes:
        source_path (str): Path to source file
        timestamps (str): Path to timestamps file
    """

    def __init__(self,g_pool,source_path=None,timestamps=None,timed_playback=False):
        super().__init__(g_pool)

        # minimal attribute set
        self._initialised = True
        self.slowdown     = 0.0
        self.source_path  = source_path
        self.timestamps   = None
        self.timed_playback = timed_playback

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

        self.display_time = 0.
        self.target_frame_idx = 0
        self.current_frame_idx = 0

        #we will use below for av playback
        # self.selected_streams = [s for s in (self.video_stream,self.audio_stream) if s]
        # self.av_packet_iterator = self.container.demux(self.selected_streams)

        if float(self.video_stream.average_rate)%1 != 0.0:
            logger.error('Videofile pts are not evenly spaced, pts to index conversion may fail and be inconsitent.')

        #load/generate timestamps.
        if timestamps is None:
            timestamps_path,ext =  os.path.splitext(source_path)
            timestamps = timestamps_path+'_timestamps.npy'
            try:
                self.timestamps = np.load(timestamps).tolist()
            except IOError:
                logger.warning("did not find timestamps file, making timetamps up based on fps and frame count. Frame count and timestamps are not accurate!")
                frame_rate = float(self.video_stream.average_rate)
                self.timestamps = [i/frame_rate for i in range(int(self.container.duration/av.time_base*frame_rate)+100)] # we are adding some slack.
            else:
                logger.debug("Auto loaded %s timestamps from %s"%(len(self.timestamps),timestamps))
        else:
            logger.debug('using timestamps from list')
            self.timestamps = timestamps

        # set the pts rate to convert pts to frame index. We use videos with pts writte like indecies.
        self.next_frame = self._next_frame()
        f0, f1 = next(self.next_frame), next(self.next_frame)
        self.pts_rate = f1.pts
        self.seek_to_frame(0)
        self.average_rate = (self.timestamps[-1]-self.timestamps[0])/len(self.timestamps)

    def ensure_initialisation(fallback_func=None):
        from functools import wraps

        def decorator(func):
            @wraps(func)
            def run_func(self, *args, **kwargs):
                if self._initialised and self.video_stream:
                    return func(self, *args, **kwargs)
                elif fallback_func:
                    return fallback_func(*args, **kwargs)
                else:
                    logger.debug('Initialisation required.')
            return run_func
        return decorator

    @property
    def initialised(self):
        return self._initialised

    @property
    def frame_size(self):
        return int(self.video_stream.format.width), int(self.video_stream.format.height)

    @property
    @ensure_initialisation(fallback_func=lambda: 20)
    def frame_rate(self):
        return float(self.average_rate)

    def get_init_dict(self):
        settings = super().get_init_dict()
        settings['source_path'] = self.source_path
        settings['timestamps'] = self.timestamps
        settings['timed_playback'] = self.timed_playback
        return settings

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
        raise EndofVideoFileError("end of file.")

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
                # print 'skip frame to seek','now at:',index
            else:
                logger.debug('Frame index not consistent.')
                break
        if not frame:
            raise EndofVideoFileError('Reached end of videofile')

        try:
            timestamp = self.timestamps[index]
        except IndexError:
            logger.warning("Reached end of timestamps list.")
            raise EndofVideoFileError("Reached end of timestamps list.")

        self.show_time = timestamp
        self.target_frame_idx = index+1
        self.current_frame_idx = index
        return Frame(timestamp,frame,index=index)

    def wait(self,frame):
        if self.display_time:
            wait_time  = frame.timestamp - self.display_time - time()
            if 1 > wait_time > 0 :
                sleep(wait_time)
        self.display_time = frame.timestamp - time()
        sleep(self.slowdown)

    @ensure_initialisation(fallback_func=lambda evt: sleep(0.05))
    def recent_events(self,events):
        try:
            frame = self.get_frame()
        except EndofVideoFileError:
            logger.info('Video has ended.')
            self._initialised = False
        else:
            self._recent_frame = frame
            events['frame'] = frame
            if self.timed_playback:
                self.wait(frame)

    @ensure_initialisation()
    def seek_to_frame(self, seek_pos):
        ###frame accurate seeking
        try:
            self.video_stream.seek(self.idx_to_pts(seek_pos),mode='time')
        except av.AVError as e:
            raise FileSeekError()
        else:
            self.next_frame = self._next_frame()
            self.display_time = 0
            self.target_frame_idx = seek_pos

    @ensure_initialisation()
    def seek_to_frame_fast(self, seek_pos):
        ###frame accurate seeking
        try:
            self.video_stream.seek(self.idx_to_pts(seek_pos), mode='time', any_frame=True)
        except av.AVError as e:
            raise FileSeekError()
        else:
            self.next_frame = self._next_frame()
            self.display_time = 0
            self.target_frame_idx = seek_pos

    def on_notify(self, notification):
        if notification['subject'] == 'file_source.seek' and notification.get('source_path') == self.source_path:
            self.seek_to_frame(notification['frame_index'])

    def init_gui(self):
        from pyglui import ui
        ui_elements = []
        ui_elements.append(ui.Info_Text("Running Capture with '%s' as src"%self.source_path))
        ui_elements.append(ui.Slider('slowdown',self,min=0,max=1.0))
        self.g_pool.capture_source_menu.extend(ui_elements)

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
    file_exts = ['.mp4','.mkv','.mov']

    def __init__(self, g_pool, root_folder=None):
        super().__init__(g_pool)
        base_dir = self.g_pool.user_dir.rsplit(os.path.sep,1)[0]
        default_rec_dir = os.path.join(base_dir,'recordings')
        self.root_folder = root_folder or default_rec_dir

    def init_gui(self):
        from pyglui import ui
        ui_elements = []
        ui_elements.append(ui.Info_Text('Enter a folder to enumerate all eligible video files. Be aware that entering folders with a lot of files can slow down Pupil Capture.'))

        def set_root(folder):
            if not os.path.isdir(folder):
                logger.error('`%s` is not a valid folder path.'%folder)
            else: self.root_folder = folder

        ui_elements.append(ui.Text_Input('root_folder',self,label='Source Folder',setter=set_root))

        def split_enumeration():
            eligible_files = self.enumerate_folder(self.root_folder)
            eligible_files.insert(0, (None, 'Select to activate'))
            return zip(*eligible_files)

        def activate(full_path):
            if not full_path:
                return
            settings = {
                'source_path': full_path,
                'timed_playback': True
            }
            self.activate_source(settings)

        ui_elements.append(ui.Selector(
            'selected_file',
            selection_getter=split_enumeration,
            getter=lambda: None,
            setter=activate,
            label='Video File'
        ))

        self.g_pool.capture_selector_menu.extend(ui_elements)

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
