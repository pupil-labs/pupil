'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
av_writer module uses PyAV (ffmpeg or libav backend) to write AV files.
requires:
    -
"""
import os,sys,platform
import av
from av.packet import Packet
import numpy as np
from time import time
from fractions import Fraction

#logging
import logging
logger = logging.getLogger(__name__)


from threading import Thread
from threading import Event



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

    def __init__(self, file_loc,fps=30, video_stream={'codec':'mpeg4','bit_rate': 15000*10e3}, audio_stream=None,use_timestamps=False):
        super(AV_Writer, self).__init__()
        self.use_timestamps = use_timestamps
        # the approximate capture rate.
        self.fps = int(fps)
        try:
            file_path,ext = file_loc.rsplit('.', 1)
        except:
            logger.error("'%s' is not a valid media file name."%file_loc)
            raise Exception("Error")

        if ext not in ('mp4,mov,mkv'):
            logger.warning("media file container should be mp4 or mov. Using a different container is risky.")

        self.ts_file_loc = file_path+'_timestamps_pts.npy'
        self.file_loc = file_loc
        self.container = av.open(self.file_loc,'w')
        logger.debug("Opended '%s' for writing."%self.file_loc)

        if self.use_timestamps:
            self.time_base = Fraction(1,65535) #highest resolution for mp4
        else:
            self.time_base = Fraction(1000,self.fps*1000) #timebase is fps

        self.video_stream = self.container.add_stream(video_stream['codec'],1/self.time_base)
        self.video_stream.bit_rate = video_stream['bit_rate']
        self.video_stream.bit_rate_tolerance = video_stream['bit_rate']/20
        self.video_stream.thread_count = 1
        # self.video_stream.pix_fmt = "yuv420p"
        self.configured = False
        self.start_time = None

        self.current_frame_idx = 0

    def write_video_frame(self, input_frame):
        if not self.configured:
            self.video_stream.height = input_frame.height
            self.video_stream.width = input_frame.width
            self.configured = True
            self.start_time = input_frame.timestamp
            if input_frame.yuv_buffer:
                self.frame = av.VideoFrame(input_frame.width, input_frame.height,'yuv422p')
            else:
                self.frame = av.VideoFrame(input_frame.width,input_frame.height,'bgr24')
            if self.use_timestamps:
                self.frame.time_base = self.time_base
            else:
                self.frame.time_base = Fraction(1,self.fps)

        if input_frame.yuv_buffer:
            y,u,v = input_frame.yuv422
            self.frame.planes[0].update(y)
            self.frame.planes[1].update(u)
            self.frame.planes[2].update(v)
        else:
            self.frame.planes[0].update(input_frame.img)

        if self.use_timestamps:
            self.frame.pts = int( (input_frame.timestamp-self.start_time)/self.time_base )
        else:
            # our timebase is 1/30  so a frame idx is the correct pts for an fps recorded video.
            self.frame.pts = self.current_frame_idx
        #send frame of to encoder
        packet = self.video_stream.encode(self.frame)
        if packet:
            self.container.mux(packet)
        self.current_frame_idx +=1



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


    def release(self):
        self.close()


class JPEG_Writer(object):
    """
    PyAV based jpeg writer.
    """

    def __init__(self, file_loc,fps=30):
        super(JPEG_Writer, self).__init__()
        # the approximate capture rate.
        self.fps = int(fps)
        self.time_base = Fraction(1000,self.fps*1000)

        try:
            file_path,ext = file_loc.rsplit('.', 1)
        except:
            logger.error("'%s' is not a valid media file name."%file_loc)
            raise Exception("Error")

        if ext not in ('mp4'):
            logger.warning("media file container should be mp4. Using a different container is risky.")

        self.file_loc = file_loc
        self.container = av.open(self.file_loc,'w')
        logger.debug("Opended '%s' for writing."%self.file_loc)

        self.video_stream = self.container.add_stream('mjpeg',1/self.time_base)
        self.video_stream.pix_fmt = "yuvj422p"
        self.configured = False
        self.frame_count = 0

        self.write_video_frame_compressed = self.write_video_frame

    def write_video_frame(self, input_frame):
        if not self.configured:
            self.video_stream.height = input_frame.height
            self.video_stream.width = input_frame.width
            self.configured = True

        packet = Packet()
        packet.payload = input_frame.jpeg_buffer
        #we are setting the packet pts manually this uses a different timebase av.frame!
        packet.dts = int(self.frame_count/self.video_stream.time_base/self.fps)
        packet.pts = int(self.frame_count/self.video_stream.time_base/self.fps)
        self.frame_count +=1
        self.container.mux(packet)


    def close(self):
        self.container.close()
        logger.debug("Closed media container")

    def release(self):
        self.close()




def format_time(time, time_base):
        if time is None:
            return 'None'
        return '%.3fs (%s or %s/%s)' % (time_base * time, time_base * time, time_base.numerator * time, time_base.denominator)


def rec_thread(file_loc, in_container, audio_src,should_close):
    # print sys.modules['av']
    # import av
    if not in_container:
        #create in container
        if platform.system() == "Darwin":
            in_container = av.open('none:%s'%audio_src,format="avfoundation")
        elif platform.system() == "Linux":
            in_container = av.open('hw:%s'%audio_src,format="alsa")

    in_stream = None

    # print len(in_container.streams), 'stream(s):'
    for i, stream in enumerate(in_container.streams):

        if stream.type == 'audio':
            # print '\t\taudio:'
            # print '\t\t\tformat:', stream.format
            # print '\t\t\tchannels: %s' % stream.channels
            in_stream = stream
            break

    if in_stream is None:
        # logger.error("No input audio stream found.")
        return

    #create out container
    out_container = av.open(file_loc,'w')
    # logger.debug("Opended '%s' for writing."%file_loc)
    out_stream =  out_container.add_stream(template = in_stream)


    for packet in in_container.demux(in_stream):
        # for frame in packet.decode():
        #     packet = out_stream.encode(frame)
        #     if packet:
        # print '%r' %packet
        # print '\tduration: %s' % format_time(packet.duration, packet.stream.time_base)
        # print '\tpts: %s' % format_time(packet.pts, packet.stream.time_base)
        # print '\tdts: %s' % format_time(packet.dts, packet.stream.time_base)
        out_container.mux(packet)
        if should_close.is_set():
            break

    out_container.close()


class Audio_Capture(object):
    """
    PyAV based audio capture.
    """

    def __init__(self, file_loc,audio_src=0):
        super(Audio_Capture, self).__init__()
        self.thread = None

        try:
            file_path,ext = file_loc.rsplit('.', 1)
        except:
            logger.error("'%s' is not a valid media file name."%file_loc)
            raise Exception("Error")

        if ext not in ('wav'):
            logger.error("media file container should be wav. Using a different container is not supported.")
            raise NotImplementedError()

        self.should_close = Event()

        self.start(file_loc,audio_src)

    def start(self,file_loc, audio_src):
        self.should_close.clear()
        if platform.system() == "Darwin":
            in_container = av.open('none:%s'%audio_src,format="avfoundation")
        else:
            in_container = None
        self.thread = Thread(target=rec_thread, args=(file_loc,in_container, audio_src,self.should_close))
        self.thread.start()


    def stop(self):
        self.should_close.set()
        self.thread.join(timeout=1)
        self.thread = None

    def close(self):
        self.stop()

    def __del__(self):
        if self.thread:
            self.stop()


def mac_pyav_hack():
    if platform.system() == "Darwin":
        try:
            av.open(':0',format="avfoundation")
        except:
            pass


# def test():

#     import os
#     import cv2
#     from video_capture import autoCreateCapture
#     logging.basicConfig(level=logging.DEBUG)

#     writer = AV_Writer(os.path.expanduser("~/Desktop/av_writer_out.mp4"))
#     # writer = cv2.VideoWriter(os.path.expanduser("~/Desktop/av_writer_out.avi"),cv2.cv.CV_FOURCC(*"DIVX"),30,(1280,720))
#     cap = autoCreateCapture(0,(1280,720))
#     frame = cap.get_frame()
#     # print writer.video_stream.time_base
#     # print writer.

#     for x in xrange(300):
#         frame = cap.get_frame()
#         writer.write_video_frame(frame)
#         # writer.write(frame.img)
#         # print writer.video_stream

#     cap.close()
#     writer.close()




if __name__ == '__main__':
    # try:
    #     from billiard import forking_enable
    #     forking_enable(0)
    # except ImportError:
    #     pass
    logging.basicConfig(level=logging.DEBUG)

    cap = Audio_Capture('test.wav','default')

    import time
    time.sleep(5)
    cap.close()
    #mic device
    exit()


    # container = av.open('hw:0',format="alsa")
    container = av.open('1:0',format="avfoundation")
    print 'container:', container
    print '\tformat:', container.format
    print '\tduration:', float(container.duration) / av.time_base
    print '\tmetadata:'
    for k, v in sorted(container.metadata.iteritems()):
        print '\t\t%s: %r' % (k, v)
    print

    print len(container.streams), 'stream(s):'
    audio_stream = None
    for i, stream in enumerate(container.streams):

        print '\t%r' % stream
        print '\t\ttime_base: %r' % stream.time_base
        print '\t\trate: %r' % stream.rate
        print '\t\tstart_time: %r' % stream.start_time
        print '\t\tduration: %s' % format_time(stream.duration, stream.time_base)
        print '\t\tbit_rate: %r' % stream.bit_rate
        print '\t\tbit_rate_tolerance: %r' % stream.bit_rate_tolerance

        if stream.type == b'audio':
            print '\t\taudio:'
            print '\t\t\tformat:', stream.format
            print '\t\t\tchannels: %s' % stream.channels
            audio_stream = stream
            break
        elif stream.type == 'container':
            print '\t\tcontainer:'
            print '\t\t\tformat:', stream.format
            print '\t\t\taverage_rate: %r' % stream.average_rate

        print '\t\tmetadata:'
        for k, v in sorted(stream.metadata.iteritems()):
            print '\t\t\t%s: %r' % (k, v)

    if not audio_stream:
        exit()
    #file contianer:

    out_container = av.open('test.wav','w')
    out_stream = out_container.add_stream(template=audio_stream)
    # out_stream.rate = 44100
    for i,packet in enumerate(container.demux(audio_stream)):
        # for frame in packet.decode():
        #     packet = out_stream.encode(frame)
        #     if packet:
        print '%r' %packet
        print '\tduration: %s' % format_time(packet.duration, packet.stream.time_base)
        print '\tpts: %s' % format_time(packet.pts, packet.stream.time_base)
        print '\tdts: %s' % format_time(packet.dts, packet.stream.time_base)
        out_container.mux(packet)
        if i >1000:
            break

    out_container.close()

    # import cProfile,subthread,os
    # cProfile.runctx("test()",{},locals(),"av_writer.pstats")
    # loc = os.path.abspath(_file__).rsplit('pupil_src', 1)
    # gprof2dot_loc = os.path.oin(loc[0], 'pupil_src', 'shared_modules','gprof2dot.py')
    # subthread.call("python "+gprof2dot_loc+" -f pstats av_writer.pstats | dot -Tpng -o av_writer.png", shell=True)
    # print "created cpu time graph for av_writer thread. Please check out the png next to the av_writer.py file"

