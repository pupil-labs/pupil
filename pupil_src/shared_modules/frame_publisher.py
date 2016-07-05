from plugin import Plugin
from pyglui import ui
import numpy as np

class Frame_Publisher(Plugin):

    def __init__(self,g_pool,format='jpeg'):
        super(Frame_Publisher,self).__init__(g_pool)
        self.format = format

    def init_gui(self):
        help_str = "Publishes frame data in different formats under the topic \"frame.world\"."
        self.menu = ui.Growing_Menu('Frame Publisher')
        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Selector('format',self,selection=["jpeg","yuv","bgr","gray"], labels=["JPEG", "YUV", "BGR", "Gray Image"],label='Format'))
        self.g_pool.sidebar.append(self.menu)

    def update(self,frame=None,events={}):
        if frame.jpeg_buffer:

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
                'width': frame.width,
                'height': frame.width,
                'index': frame.index,
                'timestamp': frame.timestamp,
                'format': self.format,
                '__raw_data__': [blob]
            }]

    def get_init_dict(self):
        return {'format':self.format}

    def close(self):
            self.alive = False

    def cleanup(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None