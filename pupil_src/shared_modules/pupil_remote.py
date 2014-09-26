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

from recorder import Recorder

import logging
logger = logging.getLogger(__name__)



class Pupil_Remote(Plugin):
    """pupil server plugin
    send messages to control Pupil Capture functions:

    'R' toggle recording
    'R rec_name' toggle recording and name new recording rec_name
    'T' set timebase to 0
    'C' start currently selected calibration
    """
    def __init__(self, g_pool, atb_pos=(10,400),on_char_fn = None):
        Plugin.__init__(self)
        self.g_pool = g_pool
        self.on_char_fn = on_char_fn
        self.order = .9 #excecute late in the plugin list.
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.address = create_string_buffer('',512)
        self.set_server(create_string_buffer("tcp://*:50020",512))

        help_str = "Pupil Remote using REQ RREP schema. "

        self._bar = atb.Bar(name = self.__class__.__name__, label='Remote',
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
        try:
            msg = self.socket.recv(flags=zmq.NOBLOCK)
        except zmq.ZMQError :
            msg = None
        if msg:
            if msg[0] == 'R':
                rec_name = msg[2:]
                if rec_name:
                    self.g_pool.rec_name = rec_name
                self.on_char_fn(None,ord('r') ) #emulate the user hitting 'r'
            elif msg == 'T':
                self.g_pool.timebase.value = self.g_pool.capure.get_now()
                logger.info("New timebase set to %s all timestamps will count from here now."%g_pool.timebase.value)
            elif msg == 'C':
                self.on_char_fn(None,ord('c') ) #emulate the user hitting 'c'

            self.socket.send("%s - confirmed"%msg)


    def close(self):
        self.alive = False

    def cleanup(self):
        """gets called when the plugin get terminated.
           either volunatily or forced.
        """
        self._bar.destroy()
        self.context.destroy()

