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
from av.packet import Packet
import numpy as np
from time import time
from fractions import Fraction
import subprocess as sp

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
Only when the header  of the stream is written stream.time_base is garanteed
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

    def __init__(self, file_loc, video_stream={'codec':'mpeg4','bit_rate': 8000*10e3}, audio_stream=None):
        super(AV_Writer, self).__init__()

        try:
            file_path,ext = file_loc.rsplit('.', 1)
        except:
            logger.error("'%s' is not a valid media file name."%file_loc)
            raise Exception("Error")

        if ext not in ('mp4,mov,mkv'):
            logger.warning("media file container should be mp4 or mov. Using a different container is risky.")

        self.ts_file_loc = file_path+'timestamps.npy'
        self.file_loc = file_loc
        self.container = av.open(self.file_loc,'w')
        logger.debug("Opended '%s' for writing."%self.file_loc)

        self.time_resolution = 1000  # time_base in milliseconds
        self.time_base = Fraction(1,self.time_resolution)


        self.video_stream = self.container.add_stream(video_stream['codec'],self.time_resolution)
        self.video_stream.bit_rate = video_stream['bit_rate']
        # self.video_stream.pix_fmt = "yuv420p"#video_stream['format']
        self.configured = False
        self.start_time = None

        self.timestamps_list = []

    def write_video_frame(self, input_frame):
        if not self.configured:
            self.video_stream.height = input_frame.img.shape[0]
            self.video_stream.width = input_frame.img.shape[1]
            self.configured = True
            self.start_time = input_frame.timestamp

        # frame from np.array
        frame = av.VideoFrame.from_ndarray(input_frame.img, format='bgr24')
        # here we create a timestamp in ms resolution to be used for the frame pts.
        # later libav will scale this to stream timebase
        frame_ts_ms = int((input_frame.timestamp-self.start_time)*self.time_resolution)
        frame.pts = frame_ts_ms
        frame.time_base = self.time_base
        # we keep a version of the timestamp counting from first frame in the codec resoltion (lowest time resolution in toolchain)
        frame_ts_s = float(frame_ts_ms)/self.time_resolution
        # we append it to our list to correlate hi-res absolute timestamps with media timstamps
        self.timestamps_list.append((input_frame.timestamp,frame_ts_s))

        #send frame of to encoder
        packet = self.video_stream.encode(frame)
        if packet:
            # print 'paket',packet.pts
            self.container.mux(packet)


    def write_video_frame_yuv422(self, input_frame):
        if not self.configured:
            self.video_stream.height = input_frame.height
            self.video_stream.width = input_frame.width
            self.configured = True
            self.start_time = input_frame.timestamp

        frame = av.VideoFrame(input_frame.width, input_frame.height,'yuv422p')
        y,u,v = input_frame.yuv422
        frame.planes[0].update(y)
        frame.planes[1].update(u)
        frame.planes[2].update(v)

        # frame = av.VideoFrame(input_frame.width, input_frame.height,'yuv420p')
        # y,u,v = input_frame.yuv420
        # frame.planes[0].update(y)
        # frame.planes[1].update(np.ascontiguousarray(u))
        # frame.planes[2].update(np.ascontiguousarray(v))
        # here we create a timestamp in ms resolution to be used for the frame pts.
        # later libav will scale this to stream timebase
        frame_ts_ms = int((input_frame.timestamp-self.start_time)*self.time_resolution)
        frame.pts = frame_ts_ms
        frame.time_base = self.time_base
        # we keep a version of the timestamp counting from first frame in the codec resoltion (lowest time resolution in toolchain)
        frame_ts_s = float(frame_ts_ms)/self.time_resolution
        # we append it to our list to correlate hi-res absolute timestamps with media timstamps
        self.timestamps_list.append((input_frame.timestamp,frame_ts_s))

        #send frame of to encoder
        packet = self.video_stream.encode(frame)
        if packet:
            # print 'paket',packet.pts
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
        logger.debug("Closed media container")

        ts_array = np.array(self.timestamps_list)
        np.save(self.ts_file_loc,ts_array)
        logger.debug("Saved %s frames"%ts_array.shape[0])

    def release(self):
        self.close()


class JPEG_Writer(object):
    """
    Does not work yet.
    """

    def __init__(self, file_loc):
        super(JPEG_Writer, self).__init__()

        try:
            file_path,ext = file_loc.rsplit('.', 1)
        except:
            logger.error("'%s' is not a valid media file name."%file_loc)
            raise Exception("Error")

        if ext not in ('mp4,mov,mkv'):
            logger.warning("media file container should be mp4 or mov. Using a different container is risky.")

        self.file_loc = file_loc
        self.container = av.open(self.file_loc,'w')
        logger.debug("Opended '%s' for writing."%self.file_loc)



        self.video_stream = self.container.add_stream('mjpeg',1000)
        self.video_stream.pix_fmt = "yuv422p"
        print self.video_stream
        print self.container
        self.configured = False


    def write_video_frame(self, input_frame):
        if not self.configured:
            self.video_stream.height = input_frame.height
            self.video_stream.width = input_frame.width
            self.configured = True

        packet = Packet()
        packet.payload = input_frame.jpeg_buffer.view()
        self.container.mux(packet)


    def close(self):

        self.container.close()
        logger.debug("Closed media container")

    def release(self):
        self.close()


class JPEG_Dumper(object):
    """simple for JPEG_Dumper"""
    def __init__(self, file_loc):
        super(JPEG_Dumper, self).__init__()

        try:
            file_path,ext = file_loc.rsplit('.', 1)
        except:
            logger.error("'%s' is not a valid media file name."%file_loc)
            raise Exception("Error")

        self.raw_path = file_path+'.raw'
        self.out_path = file_loc

        self.file_handle = open(self.raw_path, 'wb')


    def write_video_frame(self,frame):
        self.file_handle.write(frame.jpeg_buffer.view())

    def release(self):
        self.file_handle.close()
        try:
            sp.Popen('ffmpeg',stdout=open(os.devnull, 'wb'),stderr=open(os.devnull, 'wb'))
        except OSError:
            logger.error("Please install ffmpeg to enable pupil capture to convert raw jpeg streams to a readable format.")
        else:
            # ffmpeg  -f mjpeg -i world.raw -vcodec copy world.mkv
            sp.call(['ffmpeg -f mjpeg -i '+self.raw_path +' -vcodec copy '+self.out_path],shell=True)
            #this should be done programatically but requires a better video backend.
            sp.Popen(["rm "+ self.raw_path],shell=True)


def ffmpeg_available():
    import platform
    if platform.system() == 'Darwin' and getattr(sys, 'frozen', False):
        return False
    try:
        sp.Popen('ffmpeg',stdout=open(os.devnull, 'wb'),stderr=open(os.devnull, 'wb'))
    except OSError:
        logger.error("Please install ffmpeg to enable pupil capture to capture raw jpeg streams as a readable format.")
        return False
    else:
        return True


def test():

    import os
    import cv2
    from video_capture import autoCreateCapture
    logging.basicConfig(level=logging.DEBUG)

    writer = AV_Writer(os.path.expanduser("~/Desktop/av_writer_out.mp4"))
    # writer = cv2.VideoWriter(os.path.expanduser("~/Desktop/av_writer_out.avi"),cv2.cv.CV_FOURCC(*"DIVX"),30,(1280,720))
    cap = autoCreateCapture(0,(1280,720))
    frame = cap.get_frame()
    # print writer.video_stream.time_base
    # print writer.

    for x in xrange(300):
        frame = cap.get_frame()
        writer.write_video_frame(frame)
        # writer.write(frame.img)
        # print writer.video_stream

    cap.close()
    writer.close()




if __name__ == '__main__':

    import cProfile,subprocess,os
    cProfile.runctx("test()",{},locals(),"av_writer.pstats")
    loc = os.path.abspath(__file__).rsplit('pupil_src', 1)
    gprof2dot_loc = os.path.join(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    subprocess.call("python "+gprof2dot_loc+" -f pstats av_writer.pstats | dot -Tpng -o av_writer.png", shell=True)
    print "created cpu time graph for av_writer process. Please check out the png next to the av_writer.py file"

