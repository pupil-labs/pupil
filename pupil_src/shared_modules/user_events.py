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

class User_Event_Capture(Plugin):
    """Describe your plugin here
    """
    def __init__(self,g_pool,events=[('My event','e')]):
        super(User_Event_Capture, self).__init__(g_pool)
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
        event = {'subject':'local_user_event','user_event_name':event_name,'timestamp':t}
        self.notify_all(event)
        if self.recording:
            self.event_list.append( event )

    def on_notify(self,notification):
        if notification['subject'] is 'rec_started':
            self.file_path = os.path.join(notification['rec_path'],'user_events')
            self.recording = True
            self.event_list = []

        elif notification['subject'] is 'rec_stopped':
            self.stop()

        elif notification['subject'] is 'remote_user_event':
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




class User_Event_Player(Plugin):
    """Describe your plugin here
    When captured file is played, event tags (straight line pertruding
         from bar) should appear along the video bar
    """
    def __init__(self,g_pool,events=[('My event','e')]):
        super(Event_Player, self).__init__(g_pool)

        self.menu = None
        self.frame_count = g_pool.capture.get_frame_index()

        #display layout
        self.padding = 20. #in sceen pixel
        self.window_size = 0,0


    def get_index(self):
        captured_events = load_object(os.path.join(self.g_pool.rec_dir, "user_addition_events"))
        dict_captured_events = [{'timestamp':i[1]} for i in captured_events]
        np_timestamps = np.load(os.path.join(self.g_pool.rec_dir, "world_timestamps.npy"))
        timestamps = np_timestamps.tolist()
        
        data = correlate_data(dict_captured_events, timestamps)

        return data


    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Event Player')

        # add menu to the window
        self.g_pool.gui.append(self.menu)

        self.menu.append(ui.Button('remove',self.unset_alive))

        self.on_window_resize(glfwGetCurrentContext(),*glfwGetWindowSize(glfwGetCurrentContext()))


    def on_window_resize(self,window,w,h):
        self.window_size = w,h
        self.h_pad = self.padding * self.frame_count/float(w)
        self.v_pad = self.padding * 1./h


    def update(self,frame,events):
        pass


    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None


    def unset_alive(self):
        self.alive = False


    def gl_display(self):

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(-self.h_pad, (self.frame_count)+self.h_pad, -self.v_pad, 1+self.v_pad,-1,1) # ranging from 0 to cache_len-1 (horizontal) and 0 to 1 (vertical)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        event_tick_color = (1.,0.,0.,.8)
        events_in_frame = self.get_index()

        for frame in events_in_frame:
            logger.info(frame)
            if len(frame) != 0:
                draw_polyline(verts=[(frame[0]['index'],0),(frame[0]['index'],20)],
                    color=RGBA(*event_tick_color))


        # draw_polyline(verts=[(0,0),(self.current_frame_index,0)],color=RGBA(*color1))
        # draw_polyline(verts=[(self.current_frame_index,0),(self.frame_count,0)],color=RGBA(.5,.5,.5,.5))
        # draw_points([(self.current_frame_index,0)],color=RGBA(*color1),size=40)
        # draw_points([(self.current_frame_index,0)],color=RGBA(*color2),size=10)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()


    def cleanup(self):
        """called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
