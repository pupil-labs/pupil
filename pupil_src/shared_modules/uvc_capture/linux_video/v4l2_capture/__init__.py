'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
This file contains bindings to a webcam capture module that works with v4l2 and is written in c
"""

from ctypes import *
from numpy.ctypeslib import ndpointer
import numpy as np
import os,sys
#logging
import logging
logger = logging.getLogger(__name__)

class CameraCaptureError(Exception):
    """General Exception for this module"""
    def __init__(self, arg):
        super(CameraCaptureError, self).__init__()
        self.arg = arg



if getattr(sys, 'frozen', False):
    # we are running in a |PyInstaller| bundle
    basedir = sys._MEIPASS
else:
    # we are running in a normal Python environment
    basedir = os.path.dirname(__file__)


if not getattr(sys, 'frozen', False):

    ### Run Autocompiler
    #  Binaries are not distributed instead a make file and source are in this folder
    #  Make is invoked when this module is imported or run.

    arch_64bit = sizeof(c_void_p) == 8
    if arch_64bit:
        c_flags = "CFLAGS=-m64"
    else:
        c_flags = "CFLAGS=-m32"

    from subprocess import check_output
    logger.debug("Compiling now.")
    compiler_status = check_output(["make",c_flags],cwd=basedir)
    logger.debug('Compiler status: %s'%compiler_status)
    del check_output
    logger.debug("Compiling done.")


### C-Types binary loading
dll_name = "capture.so"
dllabspath = basedir + os.path.sep + dll_name
if not os.path.isfile(dllabspath):
    raise Exception("v4l2 capture Error could not find binary.")

dll = CDLL(dllabspath)


from definitions import *

### C-Types Argtypes and Restype

dll.open_device.argtypes = [c_char_p,]
dll.open_device.restype = c_void_p
dll.close_device.argtypes = [c_void_p,]
dll.close_device.restype = c_int
dll.xioctl.argtypes = [c_int,c_int32,]
dll.xioctl.restype = c_int
dll.get_buffer.argtypes = [c_int,POINTER(v4l2_buffer)]
dll.get_buffer.restype = c_void_p
dll.release_buffer.restype = c_int
dll.get_time_monotonic.argtypes=[]
dll.get_time_monotonic.restype = c_double
# dll.init_device.argtypes = [c_int,POINTER(c_uint32),POINTER(c_uint32),POINTER(c_uint32),POINTER(c_uint32)]


# open device
# setup fmt
# create and map buffers
# qbuffers and start stream
# dque buffer and return POINTER
# qbuffer repead above
# stop stream
# umap buffer



def enum_formats(device):
    fmt = v4l2_fmtdesc()
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    formats = []
    while dll.xioctl(device,VIDIOC_ENUM_FMT,byref(fmt))>=0:
        formats.append( [fourcc_string(fmt.pixelformat),] )
        fmt.index += 1
    return formats

def enum_sizes(device,format):
    frmsize = v4l2_frmsizeenum()
    frmsize.pixel_format = format;
    frmsize.index = 0
    sizes = []
    while dll.xioctl(device, VIDIOC_ENUM_FRAMESIZES, byref(frmsize)) >= 0:
        if frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE:
            sizes.append((frmsize.discrete.width,frmsize.discrete.height))
        elif frmsize.type == V4L2_FRMSIZE_TYPE_STEPWISE:
            sizes.append((frmsize.stepwise.max_width, frmsize.stepwise.max_height))
        frmsize.index+=1
    return sizes

def enum_rates_dict(device,format,size):
    interval = v4l2_frmivalenum()
    interval.pixel_format = format
    interval.width,interval.height = size
    dll.xioctl(device,VIDIOC_ENUM_FRAMEINTERVALS,byref(interval))
    rates = {}
    if interval.type == V4L2_FRMIVAL_TYPE_DISCRETE:
        while dll.xioctl(device, VIDIOC_ENUM_FRAMEINTERVALS, byref(interval)) >= 0:
            rates[str(float(interval.discrete.denominator)/interval.discrete.numerator)] = (interval.discrete.denominator, interval.discrete.numerator)
            interval.index += 1
    #non-discreete rates are very seldom, the second and third case should never happen
    if interval.type == V4L2_FRMIVAL_TYPE_STEPWISE or interval.type == V4L2_FRMIVAL_TYPE_CONTINUOUS:
        minval = float(interval.stepwise.min.numerator)/interval.stepwise.min.denominator
        maxval = float(interval.stepwise.max.numerator)/interval.stepwise.max.denominator
        if interval.type == V4L2_FRMIVAL_TYPE_CONTINUOUS:
            stepval = 1
        else:
            stepval = float(interval.stepwise.step.numerator)/interval.stepwise.step.denominator
        rates = range(minval,maxval,stepval)
    return rates

def enum_rates(device,format,size):
    interval = v4l2_frmivalenum()
    interval.pixel_format = format
    interval.width,interval.height = size
    dll.xioctl(device,VIDIOC_ENUM_FRAMEINTERVALS,byref(interval))
    rates = []
    if interval.type == V4L2_FRMIVAL_TYPE_DISCRETE:
        while dll.xioctl(device, VIDIOC_ENUM_FRAMEINTERVALS, byref(interval)) >= 0:
            rates.append((interval.discrete.numerator,interval.discrete.denominator))
            interval.index += 1
    #non-discreete rates are very seldom, the second and third case should never happen
    if interval.type == V4L2_FRMIVAL_TYPE_STEPWISE or interval.type == V4L2_FRMIVAL_TYPE_CONTINUOUS:
        minval = float(interval.stepwise.min.numerator)/interval.stepwise.min.denominator
        maxval = float(interval.stepwise.max.numerator)/interval.stepwise.max.denominator
        if interval.type == V4L2_FRMIVAL_TYPE_CONTINUOUS:
            stepval = 1
        else:
            stepval = float(interval.stepwise.step.numerator)/interval.stepwise.step.denominator
        rates = range(minval,maxval,stepval)
    return rates

def fourcc_string(i):
    s = chr(i & 255)
    for shift in (8,16,24):
        s += chr(i>>shift & 255)
    return s


# class deinitions
class Frame(object):
    """docstring for Frame"""
    def __init__(self, timestamp,img,compressed_img=None, compressed_pix_fmt=None):
        self.timestamp = timestamp
        self.img = img
        self.compressed_img = compressed_img
        self.compressed_pix_fmt = compressed_pix_fmt


class VideoCapture(object):
    """docstring for v4l2_capture"""
    def __init__(self, src_id,size=(1280,720),fps=30,timebase=None,use_hw_timestamps=False):
        if src_id not in range(100):
            raise Exception("V4L2 Capture src_id not a number in 0-99")
        self.src_str = "/dev/video"+str(int(src_id))
        self.open = False
        self.initialized = False
        self.streaming = False
        self.device = -1
        self.use_hw_timestamps = use_hw_timestamps
        if timebase == None:
            logger.debug("Capture will run with default system timebase")
            self.timebase = c_double(0)
        elif isinstance(timebase,c_double):
            logger.debug("Capture will run with app wide adjustable timebase")
            self.timebase = timebase
        else:
            logger.error("Invalid timebase variable type. Will use default system timebase")
            self.timebase = c_double(0)


        self._open()
        self._verify()

        self.formats = enum_formats(self.device)
        logger.debug("Formats exposed by %s: %s"%(self.formats,self.src_str))
        if ["MJPG"] in self.formats:
            self.prefered_format =  "MJPG"
        elif ["YUYV"] in self.formats:
            self.prefered_format =  "YUYV"
        else:
            logger.warning('Camera does not support the usual img transport formats.')
            self.prefered_format = self.formats[0]
        logger.debug('Fromat choosen: %s'%self.prefered_format)

        #camera settings in v4l2 structures
        self.v4l2_format = v4l2_format()
        self.v4l2_format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        self.v4l2_format.fmt.pix.width       = size[0]
        self.v4l2_format.fmt.pix.height      = size[1]
        self.v4l2_format.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24
        self.v4l2_format.fmt.pix.field       = V4L2_FIELD_ANY
        if (-1 == dll.xioctl(self.device, VIDIOC_S_FMT, byref(self.v4l2_format))):
            self._close()
            raise CameraCaptureError("Could not set v4l2 format")
        if (-1 == dll.xioctl(self.device, VIDIOC_G_FMT, byref(self.v4l2_format))):
            self._close()
            raise CameraCaptureError("Could not get v4l2 format")
        logger.info("Size on %s: %ix%i" %(self.src_str,self.v4l2_format.fmt.pix.width,self.v4l2_format.fmt.pix.height))

        self.v4l2_streamparm = v4l2_streamparm()
        self.v4l2_streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        self.v4l2_streamparm.parm.capture.timeperframe.numerator = 1
        self.v4l2_streamparm.parm.capture.timeperframe.denominator = int(fps)
        if (-1 == dll.xioctl(self.device, VIDIOC_S_PARM, byref(self.v4l2_streamparm))):
            self._close()
            raise CameraCaptureError("Could not set v4l2 parameters")
        if (-1 == dll.xioctl(self.device, VIDIOC_G_PARM, byref(self.v4l2_streamparm))):
            self._close()
            raise CameraCaptureError("Could not get v4l2 parameters")
        logger.info("Framerate on %s: %i/%i" %(self.src_str,self.v4l2_streamparm.parm.capture.timeperframe.numerator,\
                                                      self.v4l2_streamparm.parm.capture.timeperframe.denominator))

        #structure for atb menue
        size = self.v4l2_format.fmt.pix.width,self.v4l2_format.fmt.pix.height
        self.sizes = enum_sizes(self.device,v4l2_fourcc(*self.prefered_format))
        logger.debug("Sizes avaible on %s %s"%(self.src_str,self.sizes))
        self.rates = enum_rates(self.device,v4l2_fourcc(*self.prefered_format),size)
        logger.debug("Rates avaible on %s @ %s: %s"%(self.src_str,size,self.rates))
        self.sizes_menu = dict(zip([str(w)+"x"+str(h) for w,h in self.sizes], range(len(self.sizes))))
        try:
            self.current_size_idx = self.sizes.index(size)
        except ValueError:
            logger.warning("Buggy Video Camera: Not all available sizes are exposed.")
            self.current_size_idx = 0

        #structure for atb menue
        self.rates = enum_rates(self.device,v4l2_fourcc(*self.prefered_format),size)
        self.rates_menu = dict(zip([str(float(d)/n) for n,d in self.rates], range(len(self.rates))))
        fps = self.v4l2_streamparm.parm.capture.timeperframe.numerator,\
              self.v4l2_streamparm.parm.capture.timeperframe.denominator
        try:
            self.current_rate_idx = self.rates.index(fps)
        except ValueError:
            logger.warning("Buggy Video Camera: Not all available rates are exposed.")
            self.current_rate_idx = 0

        self._init()
        self._start()

        self._active_buffer = None


    def get_size(self):
        return self.sizes[self.current_size_idx]


    def get_rate(self):
        n,d = self.rates[self.current_rate_idx]
        return int(float(d)/n)


    def set_rate_idx(self,rate_id):
        new_rate = self.rates[rate_id]
        self.set_rate(new_rate)

    def set_rate(self,new_rate):
        self._stop()
        self._uninit()

        self.v4l2_streamparm = v4l2_streamparm()
        self.v4l2_streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        self.v4l2_streamparm.parm.capture.timeperframe.numerator = new_rate[0]
        self.v4l2_streamparm.parm.capture.timeperframe.denominator =  new_rate[1]
        if (-1 == dll.xioctl(self.device, VIDIOC_S_PARM, byref(self.v4l2_streamparm))):
            self._close()
            raise CameraCaptureError("Could not set v4l2 parameters")
        if (-1 == dll.xioctl(self.device, VIDIOC_G_PARM, byref(self.v4l2_streamparm))):
            self._close()
            raise CameraCaptureError("Could not get v4l2 parameters")
        logger.info("Framerate on %s: %i/%i" %(self.src_str,self.v4l2_streamparm.parm.capture.timeperframe.numerator, \
                                                      self.v4l2_streamparm.parm.capture.timeperframe.denominator))

        #update for atb menue
        fps = self.v4l2_streamparm.parm.capture.timeperframe.numerator,\
              self.v4l2_streamparm.parm.capture.timeperframe.denominator
        self.current_rate_idx = self.rates.index(fps)

        self._init()
        self._start()



    def _open(self):
        self.device = dll.open_device(c_char_p(self.src_str))
        if self.device is not -1:
            self.open = True
        else:
            raise CameraCaptureError("Capture Error: Could not open device at %s" %self.src_str)

    def _verify(self):
        if self.open:
            dll.verify_device(self.device)

    def _init(self):
        if self.open:
            dll.init_mmap(self.device)
            self.initialized = True
        else:
            raise CameraCaptureError("Capture Error: You need to open the device first")

    def _start(self):
        if self.initialized:
            dll.start_capturing(self.device)
            self.streaming = True
        else:
            self._close()
            raise CameraCaptureError("Capture Error: device is not initialized %s" %self.src_str)

    def read(self,retry=3):
        if not retry:
            self._stop()
            self._uninit()
            self._close()
            raise CameraCaptureError("Capture Error: Could not communicate with camera at: %s. Attach each camera to a single USB Controller, this may solve the problem." %self.src_str)

        if self._active_buffer:
            if not dll.release_buffer(self.device,byref(self._active_buffer)):
                logger.error("Could not release buffer VIDEOC_QBUF_ERROR")

        buf  = v4l2_buffer()
        buf_ptr =  dll.get_buffer(self.device,byref(buf))

        #did the camera return a frame at all?
        if buf_ptr:
            self._active_buffer = buf
            #is the frame ok?
            if not buf.flags & V4L2_BUF_FLAG_ERROR:

                # if buf.flags & V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC & V4L2_BUF_FLAG_TIMESTAMP_MASK:
                #     print "monotonic timebase"
                # if buf.flags & V4L2_BUF_FLAG_TSTAMP_SRC_MASK & V4L2_BUF_FLAG_TSTAMP_SRC_SOE:
                #     print "hardware timestamp"
                # logger.debug("buffer timestamp monotonic")
                if self.use_hw_timestamps:
                    timestamp = buf.timestamp.secs+buf.timestamp.usecs/1000000.
                else:
                    timestamp = self.get_time_monotonic()
                    
                timestamp -= self.timebase.value
                # if ( self.ts > timestamp) or 1:
                #     print "%s %s" %(self.src_str, self.get_time_monotonic()-timestamp)
                buf_ptr = cast(buf_ptr,POINTER(c_uint8*buf.bytesused))
                img = np.frombuffer(buf_ptr.contents,c_uint8)
                img.shape = (self.v4l2_format.fmt.pix.height,self.v4l2_format.fmt.pix.width,3)
                return Frame(timestamp, img)
            else:
                logger.warning("Frame corrupted skipping it")
                return self.read()
        else:
            logger.warning("Failed to retrieve frame from %s , Retrying"%self.src_str)
            self._active_buffer = None
            return self.read(retry-1)

    def _stop(self):
        if self.streaming:
            if self._active_buffer:
                if not dll.release_buffer(self.device,byref(self._active_buffer)):
                    logger.Error("Could not release buffer")
                self._active_buffer = None

            if not dll.stop_capturing(self.device):
                logger.error("Device not found. Could not stop it.")
            self.streaming = False

    def get_time_monotonic(self):
        return dll.get_time_monotonic()

    def _uninit(self):
        if self.initialized:
            dll.uninit_device(self.device)
            self.initialized = False


    def _close(self):
        if self.open:
            self.device = dll.close_device(self.device)
            if self.device == 0:
                logger.error("Could not close device: %s" %self.src_str)
            else:
                self.open=False
                logger.info("Closed: %s" %self.src_str)

    def cleanup(self):
        self._stop()
        self._uninit()
        self._close()

    def __del__(self):
        self._stop()
        self._uninit()
        self._close()


### Debugging
if __name__ == '__main__' :
    import numpy as np
    import cv2
    from time import sleep
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)


    # cap = cv2.VideoCapture(0)
    # cap.set(3,1920)
    # cap.set(4,1080)
    # print cap.get(3)
    # for x in range(900):
    #     s,img = cap.read()
    # device =  dll.open_device(c_char_p("/dev/video0"))
    # width,height,fps = c_uint32(1920),c_uint32(1080),c_uint32(30)
    # # dll.enum_frameformats(device)
    # # dll.init_device(device,width,height,fps)
    # print enum_formats(device)
    # print enum_sizes(device,v4l2_fourcc(*'MJPG'))
    # print enum_rates(device,v4l2_fourcc(*'MJPG'),(1280,720))
    # dll.init_device(device,width,height,fps)
    # dll.start_capturing(device)
    # for x in range(300):
    # #     # s= img.copy()
    # #     # np.random.random((1920,1080,3))
    # #     # dll.mainloop(device)
    #     buf  = v4l2_buffer()
    #     buf_ptr =  dll.get_buffer(device,byref(buf))
    #     buf_ptr = cast(buf_ptr,POINTER(c_uint8*buf.bytesused))
    # #     a = np.frombuffer(buf_ptr.contents,c_uint8)
    # #     a.shape = (height.value,width.value,3)
    # #     # print a.shape
    # #     # print buf.index
    # #     # sleep(1)
    # #     # print a[:9]
    # #     # np.save("img.npy",a)
    # #     # b = a.copy()
    # #     # del a
    # #     # del buf_ptr
    #     dll.release_buffer(device,byref(buf))
    # print "stopping"
    # dll.stop_capturing(device)
    # dll.uninit_device(device)
    # dll.close_device(device)
    # dll.fprintf(stderr, "\n")
    cap = VideoCapture(2,(320,240),30)

    for x in range(100):
        frame = cap.read()
        # print frame.img.shape
        # prin?t frame.timestamp
    # cap.set_rate(1)

    # for x in range(30):
    #     frame = cap.read()
    #     print frame.img.shape
    del cap