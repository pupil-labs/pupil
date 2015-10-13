'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os,sys
import av

assert av.__version__ >= '0.2.5'

import numpy as np
from time import time,sleep
from fractions import Fraction
from  multiprocessing import cpu_count
#logging
import logging
logger = logging.getLogger(__name__)

#UI
from pyglui import ui

class FileCaptureError(Exception):
    """General Exception for this module"""
    def __init__(self, arg):
        super(FileCaptureError, self).__init__()
        self.arg = arg

class EndofVideoFileError(Exception):
    """docstring for EndofVideoFileError"""
    def __init__(self, arg):
        super(EndofVideoFileError, self).__init__()
        self.arg = arg


class FileSeekError(Exception):
    """docstring for EndofVideoFileError"""
    def __init__(self):
        super(FileSeekError, self).__init__()


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
        if self._img is None:
            self._img = self._av_frame.to_nd_array(format='bgr24')
        return self._img

    @property
    def gray(self):
        if self._gray is None:
            self._gray = np.frombuffer(self._av_frame.planes[0],np.uint8).reshape(self.height,self.width)
        return self._gray



class File_Capture(object):
    """
    simple file capture.
    """
    def __init__(self,src,timestamps=None):
        self.menu = None
        self.display_time = 0.
        self.target_frame_idx = 0

        assert os.path.isfile(src)
        self.src = src

        self.container = av.open(src)

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

        #we will use below for av playback
        # self.selected_streams = [s for s in (self.video_stream,self.audio_stream) if s]
        # self.av_packet_iterator = self.container.demux(self.selected_streams)

        if float(self.video_stream.average_rate)%1 != 0.0:
            logger.error('Videofile pts are not evenly spaced, pts to index conversion may fail and be inconsitent.')

        #load/generate timestamps.
        if timestamps is None:
            timestamps_path,ext =  os.path.splitext(src)
            timestamps = timestamps_path+'_timestamps.npy'
            try:
                self.timestamps = np.load(timestamps).tolist()
            except IOError:
                logger.warning("did not find timestamps file, making timetamps up based on fps and frame count. Frame count and timestamps are not accurate!")
                frame_rate = float(self.video_stream.average_rate)
                self.timestamps = [i/frame_rate for i in xrange(int(self.container.duration/av.time_base*frame_rate)+100)] # we are adding some slack.
            else:
                logger.debug("Auto loaded %s timestamps from %s"%(len(self.timestamps),timestamps))
        else:
            logger.debug('using timestamps from list')
            self.timestamps = timestamps
        self.next_frame = self._next_frame()

    @property
    def name(self):
        return 'File Capture'


    @property
    def frame_size(self):
        if self.video_stream:
            return int(self.video_stream.format.width),int(self.video_stream.format.height)
        else:
            logger.error("No videostream.")

    @property
    def frame_rate(self):
        return self.video_stream.average_rate

    @property
    def settings(self):
        logger.warning("File capture has no settings.")
        return {}

    @settings.setter
    def settings(self,settings):
        logger.warning("File capture ignores settings.")


    def get_frame_index(self):
        return self.target_frame_idx

    def get_frame_count(self):
        return len(self.timestamps)

    def _next_frame(self):
        for packet in self.container.demux(self.video_stream):
            for frame in packet.decode():
                if frame:
                    yield frame
        raise EndofVideoFileError("end of file.")

    def pts_to_idx(self,pts):
        # some older mkv did not use perfect timestamping so we are doing int(round()) to clear that.
        # With properly spaced pts (any v0.6.100+ recording) just int() would suffice.
        # print float(pts*self.video_stream.time_base*self.video_stream.average_rate),round(pts*self.video_stream.time_base*self.video_stream.average_rate)
        return int(round(pts*self.video_stream.time_base*self.video_stream.average_rate))

    def pts_to_time(self,pts):
        ### we do not use this one, since we have our timestamps list.
        return int(pts*self.video_stream.time_base)

    def idx_to_pts(self,idx):
        return int(idx/self.video_stream.average_rate/self.video_stream.time_base)

    def get_frame_nowait(self):
        frame = None
        for frame in self.next_frame:
            index = self.pts_to_idx(frame.pts)
            if index == self.target_frame_idx:
                break
            elif index < self.target_frame_idx:
                pass
                # print 'skip frame to seek','now at:',index
            else:
                logger.error('Frame index not consistent.')
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
        return Frame(timestamp,frame,index=index)

    def wait(self,frame):
        if self.display_time:
            wait_time  = frame.timestamp - self.display_time - time()
            if 1 > wait_time > 0 :
                sleep(wait_time)
        self.display_time = frame.timestamp - time()

    def get_frame(self):
        frame = self.get_frame_nowait()
        self.wait(frame)
        return frame

    def seek_to_frame(self, seek_pos):
        ###frame accurate seeking
        self.video_stream.seek(self.idx_to_pts(seek_pos),mode='time')
        self.next_frame = self._next_frame()
        self.display_time = 0
        self.target_frame_idx = seek_pos

    def seek_to_frame_fast(self, seek_pos):
        ###best effort seeking to closest keyframe
        self.video_stream.seek(self.idx_to_pts(seek_pos),mode='time')
        self.next_frame = self._next_frame()
        frame = self.next_frame.next()
        index = self.pts_to_idx(frame.pts)
        self.target_frame_idx = index+1
        self.display_time = 0


    def get_now(self):
        try:
            timestamp = self.timestamps[self.get_frame_index()]
            logger.warning("Filecapture is not a realtime source. -NOW- will be the current timestamp")
        except IndexError:
            logger.warning("timestamp not found.")
            timestamp = 0
        return timestamp

    def get_timestamp(self):
        return self.get_now()

    def init_gui(self,sidebar):
        self.menu = ui.Growing_Menu(label='File Capture Settings')
        self.menu.append(ui.Info_Text("Running Capture with '%s' as src"%self.src))
        self.sidebar = sidebar
        self.sidebar.append(self.menu)

    def deinit_gui(self):
        if self.menu:
            self.sidebar.remove(self.menu)
            self.menu = None

    def close(self):
        self.deinit_gui()
if __name__ == '__main__':
    import os
    import cv2
    logging.basicConfig(level=logging.DEBUG)
    # file_loc = os.path.expanduser("~/Pupil/pupil_code/recordings/2015_09_30/019/world.mp4")
    file_loc = os.path.expanduser("~/Desktop/Marker_Tracking_Demo_Recording/world_viz.mp4")
    # file_loc = os.path.expanduser('~/pupil/recordings/2015_09_30/000/world.mp4')
    # file_loc = os.path.expanduser("~/Desktop/MAH02282.MP4")
    file_loc = os.path.expanduser("/Users/mkassner/Downloads/P012/world.mkv")

    logging.getLogger("libav").setLevel(logging.ERROR)

    cap = File_Capture(file_loc)
    print 'durantion',float(cap.container.duration/av.time_base)
    print 'frame_count', cap.get_frame_count()
    print "container timebase",av.time_base
    print 'time_base',cap.video_stream.time_base
    print 'avg rate',cap.video_stream.average_rate
    # exit()
    # print cap.video_stream.time_base
    import time
    frame = cap.get_frame_nowait()
    t = time.time()
    try:
        while 1:
            frame = cap.get_frame_nowait()
            # print frame.index
            # cv2.imshow("test",frame.img)
            # print frame.index, frame.timestamp
            # if cv2.waitKey(30)==27:
            #     print "seeking to ",95
            #     cap.seek_to_frame(95)

    except EndofVideoFileError:
        print 'Video Over'
    print'avcapture took:', time.time()-t

    import cv2
    cap = cv2.VideoCapture(file_loc)
    s,i = cap.read()

    t = time.time()
    while s:
        s,i = cap.read()
    print time.time()-t

