'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
from plugin import Plugin
from pyglui import ui
import numpy as np

class Frame_Publisher(Plugin):

    def __init__(self,g_pool,format='jpeg'):
        super().__init__(g_pool)
        self._format = format

    def init_gui(self):
        help_str = "Publishes frame data in different formats under the topic \"frame.world\"."
        self.menu = ui.Growing_Menu('Frame Publisher')
        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Selector('format',self,selection=["jpeg","yuv","bgr","gray"], labels=["JPEG", "YUV", "BGR", "Gray Image"],label='Format'))
        self.g_pool.sidebar.append(self.menu)

    def update(self,events):
        frame  = events.get("frame")
        if frame and frame.jpeg_buffer:
            if   self.format == "jpeg":
                data = frame.jpeg_buffer
            elif self.format == "yuv":
                data = frame.yuv_buffer
            elif self.format == "bgr":
                data = frame.bgr
            elif self.format == "gray":
                data = frame.gray

            # Create serializable object.
            # Not necessary if __raw_data__ key is used.
            # blob = memoryview(np.asarray(data).data)
            blob = data

            events['frame.world'] = [{
                'topic':'frame',
                'width': frame.width,
                'height': frame.height,
                'index': frame.index,
                'timestamp': frame.timestamp,
                'format': self.format,
                '__raw_data__': [blob]
            }]

    def on_notify(self,notification):
        """Publishes frame data in several formats

        Reacts to notifications:
            ``eye_process.started``: Re-emits ``frame_publishing.started``

        Emits notifications:
           ``frame_publishing.started``: Frame publishing started
           ``frame_publishing.stopped``: Frame publishing stopped
        """
        if notification['subject'].startswith('eye_process.started'):
            # trigger notification
            self.format = self.format

    def get_init_dict(self):
        return {'format':self.format}

    def close(self):
            self.alive = False

    def cleanup(self):
        self.notify_all({'subject':'frame_publishing.stopped'})
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self,value):
        self._format = value
        self.notify_all({'subject':'frame_publishing.started','format':value})