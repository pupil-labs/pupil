'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
av_writer module uses PyAV (ffmpeg or libav backend) to write AV files.
requires:
    -
"""
import os,sys
import av
import numpy as np
from time import time
from fractions import Fraction

#logging
import logging
logger = logging.getLogger(__name__)


"""
notes on time_bases and presentation timestamps:

Time_base (fraction) is the conversion factor to multipy the uint64 pts value to seconds

two time bases that we care about exsist:
    time_base of the stream (AVStream) this is used for the packet pts/dts
    time_base fo the codec (AVCodecContext) this is used for the frame

going from packet pts to frame pts when decoding:
frame.pts  = av_rescale_q ( packet. pts , packetTimeBase , frameTimeBase )

..when encoding:
packet.pts  = av_rescale_q ( frame. pts , frameTimeBase,packetTimeBase  )


Setting the time_base:
The timebase of the codec is settable (and only settable at the beginnig):
currently in PyAV this is done container.add_stream(codec,codec_timebase)

The timebase of the stream is not user settable. It is determined by ffmpeg.
The streamtimebase uses the codec timebase as a hint to find a good value.
The stream timebase in influenced by the contraints/rules of the container as well.
Only when the header header of the stream is written stream.time_base is garanteed
to be valid and should only now be accesed.





"""

class AV_Writer(object):
    """
    AV_Writer class
        - file_loc: path to file out
        - video_stream:
        - audio_stream:


    We are creating a
    """
    def __init__(self, file_loc, video_stream={'codec':'mpeg4', 'format': 'yuv420p', 'bit_rate': 5500*10e3}, audio_stream=None):
        super(AV_Writer, self).__init__()
        self.file_loc = file_loc

        time_resolution = 1000  # time_base in milliseconds
        self.time_base = Fraction(1,time_resolution)

        self.container = av.open(file_loc, 'w')
        self.video_stream = self.container.add_stream(video_stream['codec'],time_resolution)
        self.video_stream.bit_rate = video_stream['bit_rate']
        self.video_stream.pix_fmt = video_stream['format']
        self.configured = False
        self.start_time = None

    def write_video_frame(self, input_frame):
        if not self.configured:
            self.video_stream.height = input_frame.img.shape[0]
            self.video_stream.width = input_frame.img.shape[1]
            self.configured = True
            self.start_time = input_frame.timestamp

        frame = av.VideoFrame.from_ndarray(input_frame.img, format='bgr24')
        frame.pts = int((input_frame.timestamp-self.start_time)/self.time_base)
        frame.time_base = self.time_base
        print frame.pts
        print 'frame',frame.pts/self.video_stream.time_base*self.time_base
        packet = self.video_stream.encode(frame)
        if packet:
            print 'paket',packet.pts
            self.container.mux(packet)


    def close(self):
        # flush encoder
        while 1:
            packet = self.video_stream.encode()
            if packet:
                self.container.mux(packet)
            else:
                break

        self.container.close()

if __name__ == '__main__':
    import os
    import cv2
    from uvc_capture import autoCreateCapture
    logging.basicConfig(level=logging.DEBUG)

    writer = AV_Writer(os.path.expanduser("~/Desktop/av_writer_out.mp4"))
    cap = autoCreateCapture(0,(1280,720))
    frame = cap.get_frame()

    for x in xrange(900):
        frame = cap.get_frame()
        writer.write_video_frame(frame)
        # print writer.video_stream

    cap.close()
    writer.close()