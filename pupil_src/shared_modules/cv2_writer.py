'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from cv2 import VideoWriter

class CV_Writer(object):
    """docstring for CV_Writer"""
    def __init__(self, file_loc,frame_rate,frame_size):
        super().__init__()
        self.writer = VideoWriter(file_loc, VideoWriter_fourcc(*'DIVX'), float(frame_rate), frame_size)

    def write_video_frame(self, input_frame):
        self.writer.write(input_frame.img)


    def write_video_frame_yuv422(self, input_frame):
        raise NotImplementedError

    def release(self):
        self.writer.release()

