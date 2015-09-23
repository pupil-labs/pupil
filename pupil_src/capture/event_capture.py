'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''
import os
from pyglui import ui
from plugin import Plugin
from file_methods import save_object
#logging
import logging
logger = logging.getLogger(__name__)

class Event_Capture(Plugin):
    """Describe your plugin here
    """
    def __init__(self,g_pool):
        super(Event_Capture, self).__init__(g_pool)
        self.menu = None
        self.button = None

        self.recording = False
        self.events = []


    def init_gui(self):
        #lets make a menu entry in the sidebar
        self.menu = ui.Growing_Menu('My Plugin')
        self.g_pool.sidebar.append(self.menu)

        #add a button to close the plugin
        self.menu.append(ui.Button('close Event_Capture',self.close))

        #add a button to invoke event capture
        self.button = ui.Thumb('event',setter=self.add_event, getter=lambda: False, 
            label='Event Capture',hotkey='e')
        self.button.on_color[:] = (1,.0,.0,.8)
        self.g_pool.quickbar.insert(1,self.button)


    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None

    def close(self):
        self.alive = False

    def add_event(self,_):
        if self.recording:

    def on_notify(self,notification):
        if notification['name'] is 'rec_started':
            self.recording = True
            self.events = []

        elif notification['name'] is 'rec_stopped':
            self.recording = False
            rec_path = notification['rec_path']
            #do the stuff to save the data to file. Have a look at file_methods in shared_modules
            save_object(self.events,os.path.join(rec_path, "user_addition_events"))
            logger.info("added user events at: %s"%os.path.join(rec_path, "user_addition_events"))

    def get_init_dict(self):
        #anything vars we want to be persistent accross sessions need to show up in the __init__
        #and identically as a dict entry below:
        return {}


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
