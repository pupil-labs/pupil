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

class User_Events(Plugin):
    """Describe your plugin here
    """
    def __init__(self,g_pool,events=[('My event','e')]):
        super(User_Events, self).__init__(g_pool)
        self.menu = None
        self.sub_menu = None
        self.buttons = []

        self.recording = False
        self.events = events[:]
        self.event_list = []

        self.new_event_name = 'new name'
        self.new_event_hotkey = 'e'

        self.file_path = None

    def init_gui(self):
        #lets make a menu entry in the sidebar
        self.menu = ui.Growing_Menu('User Defined Events')
        self.g_pool.sidebar.append(self.menu)

        #add a button to close the plugin
        self.menu.append(ui.Button('close User_Events',self.close))
        self.menu.append(ui.Text_Input('new_event_name',self))
        self.menu.append(ui.Text_Input('new_event_hotkey',self))
        self.menu.append(ui.Button('add Event',self.add_event))
        self.sub_menu =ui.Growing_Menu('Events - click to remove')
        self.menu.append(self.sub_menu)
        self.update_buttons()


    def update_buttons(self):
        for b in self.buttons:
            self.g_pool.quickbar.remove(b)
            self.sub_menu.elements[:] = []
        self.buttons = []

        for e_name,hotkey in self.events:

            def make_fire(e_name,hotkey):
                return lambda _ : self.fire_event(e_name)

            def make_remove(e_name,hotkey):
                return lambda: self.remove_event((e_name,hotkey))

            button = ui.Thumb(e_name,setter=make_fire(e_name,hotkey), getter=lambda: False,
            label=e_name,hotkey=hotkey)
            self.buttons.append(button)
            self.g_pool.quickbar.append(button)
            self.sub_menu.append(ui.Button(e_name,make_remove(e_name,hotkey)))



    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None
        if self.buttons:
            for b in self.buttons:
                self.g_pool.quickbar.remove(b)
            self.buttons = []

    def add_event(self):
        self.events.append((self.new_event_name,self.new_event_hotkey))
        self.update_buttons()

    def remove_event(self,event):
        self.events.remove(event)
        self.update_buttons()

    def close(self):
        self.alive = False

    def fire_event(self,event_name):
        t = self.g_pool.capture.get_timestamp()
        logger.info('"%s"@%s'%(event_name,t))
        event = {'name':'local_user_event','user_event_name':event_name,'timestamp':t}
        self.notify_all(event)
        if self.recording:
            self.event_list.append( event )

    def on_notify(self,notification):
        if notification['name'] is 'rec_started':
            self.file_path = os.path.join(notification['rec_path'],'user_events')
            self.recording = True
            self.event_list = []

        elif notification['name'] is 'rec_stopped':
            self.stop()

        elif notification['name'] is 'remote_user_event':
            logger.info('"%s"@%s via sync from "%s"'%(notification['user_event_name'],notification['timestamp'],notification['sender']))
            if self.recording:
                self.event_list.append(notification)

    def get_init_dict(self):
        return {'events':self.events}

    def stop(self):
        self.recording = False
        save_object(self.event_list,self.file_path)
        logger.info("Saved %s user events to: %s"%(len(self.event_list),self.file_path))
        self.event_list = []

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
        if self.recording:
            self.stop()

