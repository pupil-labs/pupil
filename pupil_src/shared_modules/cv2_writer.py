'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from cv2 import VideoWriter
from cv2.cv import CV_FOURCC

class CV_Writer(object):
    """docstring for CV_Writer"""
    def __init__(self, file_loc,frame_rate,frame_size):
        super(CV_Writer, self).__init__()
        self.writer = VideoWriter(file_loc, CV_FOURCC(*'DIVX'), float(frame_rate), frame_size)

    def write_video_frame(self, input_frame):
        self.writer.write(input_frame.img)


    def write_video_frame_yuv422(self, input_frame):
        raise NotImplementedError

    def release(self):
        self.writer.release()

