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
from pyglui.cygl.utils import Render_Target,push_ortho,pop_ortho
import logging
from glfw import glfwGetFramebufferSize,glfwGetCurrentContext
import zmq_tools
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
import glfw

from time import time

class Log_to_Callback(logging.Handler):
    def __init__(self,cb):
        super().__init__()
        self.cb = cb
    def emit(self,record):
        self.cb(record)

def color_from_level(lvl):
    return {"CRITICAL":(.8,0,0,1),"ERROR":(1,0,0,1),"WARNING":(1.0,.8,0,1),"INFO":(1,1,1,1),"DEBUG":(1,1,1,.5),"NOTSET":(.5,.5,.5,.2)}[lvl]

def duration_from_level(lvl):
    return {"CRITICAL":3,"ERROR":2,"WARNING":1.5,"INFO":1,"DEBUG":1,"NOTSET":1}[lvl]


class Log_Display(Plugin):
    """docstring for Log_Display"""
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.rendered_log = []
        self.order = 0.3
        self.alpha = 0.0
        self.should_redraw = True

    def init_gui(self):

        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans',get_opensans_font_path())
        self.glfont.set_size(32)
        self.glfont.set_color_float((0.2,0.5,0.9,1.0))
        self.glfont.set_align_string(v_align='center',h_align='middle')

        self.window_size = glfwGetFramebufferSize(glfwGetCurrentContext())
        self.tex = Render_Target(*self.window_size)
        if self.g_pool.app == 'capture':
            self._socket = zmq_tools.Msg_Receiver(self.g_pool.zmq_ctx,self.g_pool.ipc_sub_url,('logging',))
            self._poller = zmq_tools.zmq.Poller()
            self._poller.register(self._socket.socket)
        else:
            self._socket = None
            self.log_handler = Log_to_Callback(self.on_log)
            logger = logging.getLogger()
            logger.addHandler(self.log_handler)
            self.log_handler.setLevel(logging.INFO)

    def on_log(self,record):
        if self.alpha < 1.0:
            self.rendered_log = []
            self.alpha = 0
        self.should_redraw = True
        self.rendered_log.append(record)
        self.alpha += duration_from_level(record.levelname) + len(str(record.msg))/100.
        self.rendered_log = self.rendered_log[-10:]
        self.alpha = min(self.alpha, 6.)

    def on_window_resize(self, window, w, h):
        self.window_scale = float(glfw.glfwGetFramebufferSize(window)[0] / glfw.glfwGetWindowSize(window)[0])
        self.glfont.set_size(32*self.window_scale)
        self.window_size = w, h
        self.tex.resize(*self.window_size)
        self.should_redraw = True

    def recent_events(self,events):
        if self._socket and self._socket.new_data:
            t,s = self._socket.recv()
            self.on_log(logging.makeLogRecord(s))
        self.alpha -= min(.2,events['dt'])

    def gl_display(self):
        if self.should_redraw:
            #render log content
            self.tex.push()
            push_ortho(*self.window_size)
            _,_,lineh = self.glfont.vertical_metrics()
            y = self.window_size[1]/3 - 0.5*lineh*len(self.rendered_log)
            for record in self.rendered_log:
                self.glfont.set_color_float((0.,0.,0.,1.))
                self.glfont.set_blur(10.5)
                self.glfont.draw_limited_text(self.window_size[0]/2.,y,str(record.processName.upper())+': '+str(record.msg),self.window_size[0]*0.8)
                self.glfont.set_blur(0.96)
                self.glfont.set_color_float(color_from_level(record.levelname))
                self.glfont.draw_limited_text(self.window_size[0]/2.,y,str(record.processName.upper())+': '+str(record.msg),self.window_size[0]*0.8)
                y += lineh
            pop_ortho()
            self.tex.pop()
            self.should_redraw = False

        if self.alpha > 0:
            self.tex.draw(min(1.0,self.alpha))


    def get_init_dict(self):
        return {}
