'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Plugin

import numpy as np
from pyglui import ui
import zmq



import logging
logger = logging.getLogger(__name__)



class Pupil_Server(Plugin):
    """pupil server plugin"""
    def __init__(self, g_pool,address="tcp://127.0.0.1:5000"):
        super(Pupil_Server, self).__init__(g_pool)
        self.order = .9
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.address = ''
        self.set_server(address)
        self.menu = None

        self.exclude_list = ['ellipse','pos_in_roi','major','minor','axes','angle','center']

    def init_gui(self):
        help_str = "Pupil Message server: Using ZMQ and the *Publish-Subscribe* scheme"
        self.menu = ui.Growing_Menu("Pupil Broadcast Server")
        self.menu.append(ui.TextInput('address',self,setter=self.set_server,label='Address'))
        self.g_pool.sidebar.append(self.menu)
        self.menu.collapsed = True


    def set_server(self,new_address):
        try:
            self.socket.bind(new_address)
            self.address = new_address
        except zmq.ZMQError:
            logger.error("Could not set Socket: %s"%new_address)

    def update(self,frame,recent_pupil_positions,events):
        for p in recent_pupil_positions:
            msg = "Pupil\n"
            for key,value in p.iteritems():
                if key not in self.exclude_list:
                    msg +=key+":"+str(value)+'\n'
            self.socket.send( msg )

        # for e in events:
        #     msg = 'Event'+'\n'
        #     for key,value in e.iteritems():
        #         if key not in self.exclude_list:
        #             msg +=key+":"+str(value).replace('\n','')+'\n'
        #     self.socket.send( msg )

    def close(self):
        self.alive = False


    def get_init_dict(self):
        d = {}
        d['address'] = self.address
        return d


    def cleanup(self):
        """gets called when the plugin get terminated.
           either volunatily or forced.
        """
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None
        self.context.destroy()

