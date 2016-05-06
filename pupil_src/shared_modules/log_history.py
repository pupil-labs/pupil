'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import os
from pyglui import ui
from plugin import Plugin
from file_methods import load_object,save_object

import numpy as np
from OpenGL.GL import *
from glfw import glfwGetWindowSize,glfwGetCurrentContext
from pyglui.cygl.utils import draw_polyline,RGBA
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

#logging
import logging
logger = logging.getLogger(__name__)

class Log_to_Callback(logging.Handler):
    def __init__(self,cb):
        super(Log_to_Callback, self).__init__()
        self.cb = cb

    def emit(self,record):
        self.cb(record)

class Log_History(Plugin):
    """Simple logging GUI that displays the last N messages from the logger"""
    def __init__(self, g_pool):
        super(Log_History, self).__init__(g_pool)
        self.log_messages = []
        self.order = 0.0
        self.menu = None
        self.num_messages = 50
        self.help_str = "This menu shows the last %s messages from the logger. See world.log or eye.log files for full logs." %(self.num_messages)        
    
    def init_gui(self):
        self.menu = ui.Scrolling_Menu('Log')
        self.g_pool.gui.append(self.menu)

        self.log_handler = Log_to_Callback(self.on_log)
        logger = logging.getLogger()
        logger.addHandler(self.log_handler)
        self.log_handler.setLevel(logging.INFO)
        self._update_gui()

    def _update_gui(self):
        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',self.close))        
        self.menu.append(ui.Info_Text(self.help_str))
        
        # show most recent log message at the top of the list
        for record in self.log_messages[-1:-self.num_messages:-1]:
            self.menu.append(ui.Info_Text("%s - %s" %(record.levelname, record.msg)))


    def on_log(self,record):
        self.log_messages.append(record)

        if self.menu:
            self._update_gui()
            

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu= None

    def close(self):
        self.alive = False

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()

    def get_init_dict(self):
        return {}