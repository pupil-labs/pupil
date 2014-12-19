'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import atb
import numpy as np
from gl_utils import draw_gl_polyline_norm
from ctypes import c_float,c_int,create_string_buffer

import cv2
import zmq
from plugin import Plugin




import logging
logger = logging.getLogger(__name__)



class Pupil_Server(Plugin):
    """pupil server plugin"""
    def __init__(self, g_pool, atb_pos=(10,400)):
        Plugin.__init__(self)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.address = create_string_buffer("tcp://127.0.0.1:5000",512)
        self.set_server(self.address)

        help_str = "Pupil Message server: Using ZMQ and the *Publish-Subscribe* scheme"

        self._bar = atb.Bar(name = self.__class__.__name__, label='Server',
            help=help_str, color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300,40))
        self._bar.define("valueswidth=170")
        self._bar.add_var("server address",self.address, getter=lambda:self.address, setter=self.set_server)
        self._bar.add_button("close", self.close)

        self.exclude_list = ['ellipse','pos_in_roi','major','minor','axes','angle','center']

    def set_server(self,new_address):
        try:
            self.socket.bind(new_address.value)
            self.address.value = new_address.value
        except zmq.ZMQError:
            logger.error("Could not set Socket.")

    def update(self,frame,recent_pupil_positions,events):
        for p in recent_pupil_positions:
            msg = "Pupil\n"
            for key,value in p.iteritems():
                if key not in self.exclude_list:
                    msg +=key+":"+str(value)+'\n'
            self.socket.send( msg )

        for e in events:
            msg = 'Event'+'\n'
            for key,value in e.iteritems():
                if key not in self.exclude_list:
                    msg +=key+":"+str(value).replace('\n','')+'\n'
            self.socket.send( msg )

    def close(self):
        self.alive = False

    def cleanup(self):
        """gets called when the plugin get terminated.
           either volunatily or forced.
        """
        self._bar.destroy()
        self.context.destroy()

