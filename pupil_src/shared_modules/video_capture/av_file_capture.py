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
import numpy as np
from time import time
from fractions import Fraction

#logging
import logging
logger = logging.getLogger(__name__)



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
    def __init__(self, timestamp,img,index=None,compressed_img=None, compressed_pix_fmt=None):
        self.timestamp = timestamp
        self.index = index
        self.img = img
        self.compressed_img = compressed_img
        self.compressed_pix_fmt = compressed_pix_fmt

    def copy(self):
        return Frame(self.timestamp,self.img.copy(),self.index)

class File_Capture():
    """
    simple file capture.
    """
    def __init__(self,src,timestamps=None):
        self.auto_rewind = True
        self.controls = None #No UVC controls available with file capture
        # we initialize the actual capture based on cv2.VideoCapture

        self.container = av.open(src)

        try:
            self.video_stream = next(s for s in self.container.streams if s.type=="video")# looking for the first videostream
            logger.debug("loaded videostream: %s"%self.video_stream)
            self.v_packet_iterator = self.container.demux(self.video_stream)
        except StopIteration:
            self.video_stream = None
            self.v_packet_iterator = None
            logger.error("No videostream found in media container")

        try:
            self.audio_stream = next(s for s in self.container.streams if s.type=='audio')# looking for the first audiostream
            logger.debug("loaded audiostream: %s"%self.audio_stream)
            self.a_packet_iterator = self.container.demux(self.audio_stream)
        except StopIteration:
            self.audio_stream = None
            self.a_packet_iterator = None
            logger.debug("No audiostream found in media container")

        self.selected_streams = [s for s in (self.video_stream,self.audio_stream) if s]
        self.av_packet_iterator = self.container.demux(self.selected_streams)


        if timestamps is None and src.endswith("eye.avi"):
            timestamps_loc = os.path.join(src.rsplit(os.path.sep,1)[0],'eye_timestamps.npy')
            logger.debug("trying to auto load eye_video timestamps with video at: %s"%timestamps_loc)
        else:
            timestamps_loc = timestamps
            logger.debug("trying to load supplied timestamps with video at: %s"%timestamps_loc)
        try:
            self.timestamps = np.load(timestamps_loc).tolist()
            logger.debug("loaded %s timestamps"%len(self.timestamps))
        except:
            logger.debug("did not find timestamps")
            self.timestamps = None

        self.next_frame = self._next_frame()


    @property
    def frame_size(self):
        if self.video_stream:
            return int(self.videostream.format.width),int(self.videostream.format.height)
        else:
            logger.error("No videostream.")

    @property
    def frame_rate(self):
        if self.video_stream:
            return float(self.videostream.average_rate)
        else:
            logger.error("No videostream.")

    def get_frame_index(self):
        pass
        # return int(self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))

    def get_frame_count(self):
        if self.timestamps is None:
            return cap.video_stream.frames
        return len(self.timestamps)

    def _next_frame(self):
        for packet in self.v_packet_iterator:
            # print packet
            for frame in packet.decode():
                if frame:
                    yield frame
        raise EndofVideoFileError("end of file.")

    def get_frame(self):

        frame = self.next_frame.next()
        # print
        print frame.pts/float(frame.time_base['den'])
        # print frame.ptr.coded_picture_number
        if self.timestamps:
            try:
                timestamp = self.timestamps[idx]
            except IndexError:
                logger.warning("Reached end of timestamps list.")
                raise EndofVideoFileError("Reached end of timestamps list.")
        else:
            timestamp = float(frame.pts)*Fraction(frame.time_base['num'],frame.time_base['den'])
        return Frame(timestamp,frame.to_nd_array('bgr24'),index=frame.index)

    def seek_to_frame(self, seek_pos):
        self.video_stream.seek(int(seek_pos),mode='backward')
        self.next_frame = self._next_frame()

        # frame.index = seek_pos


    def get_now(self):
        idx = int(self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        if self.timestamps:

            try:
                timestamp = self.timestamps[idx]
                logger.warning("Filecapture is not a realtime source. -NOW- will be the current timestamp")
            except IndexError:
                logger.warning("timestamps not found.")
                timestamp = 0
        else:
            logger.warning("Filecapture is not a realtime source. -NOW- will be the current time.")
            timestamp = time()
        return timestamp

    def create_atb_bar(self,pos):
        return 0,0

    def kill_atb_bar(self):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    import os
    import cv2
    logging.basicConfig(level=logging.DEBUG)
    cap = File_Capture(os.path.expanduser("~/Desktop/av_writer_out.mp4"))
    print cap.container.duration/float(av.time_base)
    print 'frame_count', cap.get_frame_count()
    # print "container timebase",av.
    print cap.video_stream.time_base
    # exit()
    # print cap.video_stream.time_base
    try:
        while 1:
            frame = cap.get_frame()
            # print frame.timestamp
            cv2.imshow("test",frame.img)
            if cv2.waitKey(30)==27:
                print "seeking to ",int(5/cap.video_stream.time_base)
                cap.seek_to_frame(int(5/cap.video_stream.time_base) )
                print cap.get_frame().timestamp
                # break
            # if x==50 or x==80:
                # cv2.waitKey(100)
            # print frame.index, frame.timestamp
    except EndofVideoFileError:
        pass
