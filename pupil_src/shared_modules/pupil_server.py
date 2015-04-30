'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

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
    def __init__(self, g_pool,address="tcp://127.0.0.1:5000",menu_conf = {'collapsed':True,'pos':(300,300),'size':(300,300)}):
        super(Pupil_Server, self).__init__(g_pool)
        self.order = .9
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.address = ''
        self.set_server(address)
        self.menu = None
        self.menu_conf = menu_conf

        self.exclude_list = ['ellipse','pos_in_roi','major','minor','axes','angle','center']

    def init_gui(self):
        help_str = "Pupil Message server: Using ZMQ and the *Publish-Subscribe* scheme"
        self.menu = ui.Growing_Menu("Pupil Broadcast Server")
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('address',self,setter=self.set_server,label='Address'))
        self.menu.append(ui.Button('Close',self.close))
        if self.g_pool.app == 'capture':
            self.menu.configuration = {'collapsed':self.menu_conf['collapsed']}
            self.g_pool.sidebar.append(self.menu)
        elif self.g_pool.app == 'player':
            self.menu.configuration = self.menu_conf

            self.g_pool.gui.append(self.menu)

    def deinit_gui(self):
        if self.menu:
            self.menu_conf = self.menu.configuration
            if self.g_pool.app == 'capture':
                self.g_pool.sidebar.remove(self.menu)
            elif self.g_pool.app == 'player':
                self.g_pool.gui.remove(self.menu)
            self.menu = None


    def set_server(self,new_address):
        try:
            self.socket.unbind(self.address)
            logger.debug('Detached from %s'%self.address)
        except:
            pass
        try:
            self.socket.bind(new_address)
            self.address = new_address
            logger.debug('Bound to %s'%self.address)

        except zmq.ZMQError as e:
            logger.error("Could not set Socket: %s. Reason: %s"%(new_address,e))

    def update(self,frame,events):
        for p in events['pupil_positions']:
            msg = "Pupil\n"
            for key,value in p.iteritems():
                if key not in self.exclude_list:
                    msg +=key+":"+str(value)+'\n'
            self.socket.send( msg )

        for g in events.get('gaze',[]):
            msg = "Gaze\n"
            for key,value in g.iteritems():
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
        if self.menu:
            d['menu_conf'] = self.menu.configuration
        else:
            d['menu_conf'] = self.menu_conf
        return d


    def cleanup(self):
        """gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        """
        self.deinit_gui()
        self.context.destroy()

