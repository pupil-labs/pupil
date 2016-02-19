'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import zmq
import json
import numpy as np

from plugin import Plugin
from pyglui import ui

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
        self.new_message_format = 1
        self.set_server(address)
        self.menu = None

        self.exclude_list = ['ellipse','pos_in_roi','major','minor','axes','angle','center']

    def init_gui(self):
        if self.g_pool.app == 'capture':
            self.menu = ui.Growing_Menu("Pupil Broadcast Server")
            self.g_pool.sidebar.append(self.menu)
        elif self.g_pool.app == 'player':
            self.menu = ui.Scrolling_Menu("Pupil Broadcast Server")
            self.g_pool.gui.append(self.menu)

        self.menu.append(ui.Button('Close',self.close))
        help_str = "Pupil Message server: Using ZMQ and the *Publish-Subscribe* scheme"
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('address',self,setter=self.set_server,label='Address'))
        self.menu.append(ui.Switch('new_message_format',self,label='Use JSON serialization'))

    def deinit_gui(self):
        if self.menu:
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
        event_types = [
            { 
                'name': 'pupil_positions',
                'topic': 'pupil',
                'message_header': 'Pupil'
            },
            {
                'name': 'gaze_positions',
                'topic': 'gaze',
                'message_header': 'Gaze'
            }
        ]
          
        for event_type in event_types:
            if self.new_message_format:
                event_data = {}
        
            for position in events.get(event_type['name'], []):
                if not self.new_message_format:
                    msg = event_type['message_header'] + '\n'
                    
                for key, value in position.iteritems():
                    if key not in self.exclude_list:
                        if self.new_message_format:
                            event_data[key] = value
                        else:
                            msg += key + ":" + str(value) + '\n'
                
                if self.new_message_format:            
                    self.socket.send_multipart((event_type['topic'], json.dumps(event_data)))
                else:
                    self.socket.send(msg)

    def close(self):
        self.alive = False

    def get_init_dict(self):
        return {'address':self.address}

    def cleanup(self):
        """gets called when the plugin get terminated.
        This happens either voluntarily or forced.
        """
        self.deinit_gui()
        self.context.destroy()

