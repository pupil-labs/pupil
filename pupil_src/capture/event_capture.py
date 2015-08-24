'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

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
        super(Example_Plugin, self).__init__(g_pool)
        # order (0-1) determines if your plugin should run before other plugins or after
        self.order = 0

        self.menu = None
        self.button = None

        self.recording = False
        self.events = []


    def init_gui(self):
        #lets make a menu entry in the sidebar
        self.menu = ui.Growing_Menu('My Plugin')
        self.g_pool.sidebar.append(self.menu)

        #add a button to close the plugin
        self.menu.append(ui.Button('close Example_Plugin',self.close))

        #add a button to invoke event capture
        self.button = ui.Thumb('event',setter=self.add_event,getter=lambda: True, label='Event Capture',hotkey='e')
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
        	self.events.append( ('my_event_name',self.g_pool.capture.get_timestamp()) )

    def on_notify(self,notification):
	    if notification is rec_started:
	        self.recording = True
	        self.events = []

	    elif notification is rec_stopped:
	        self.recording = False
	        rec_path = notification['rec_path']
	        #do the stuff to save the data to file. Have a look at file_methods in shared_modules
	        save_object(self.events,os.path.join(rec_path, "user_addition_events"))

