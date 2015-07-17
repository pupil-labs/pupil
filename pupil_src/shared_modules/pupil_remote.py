'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


from plugin import Plugin

from pyglui import ui
import zmq

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
    def __init__(self, g_pool,address="tcp://*:50020"):
        super(Pupil_Remote, self).__init__(g_pool)
        self.order = .01 #excecute first
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.address = 'not set'
        self.set_server(address)


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

    def init_gui(self):
        help_str = 'Pupil Remote using REQ RREP schemme'
        self.menu = ui.Growing_Menu('Pupil Remote')
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('address',self,setter=self.set_server,label='Address'))
        self.menu.append(ui.Button('Close',self.close))
        self.g_pool.sidebar.append(self.menu)

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None


    def update(self,frame,events):
        try:
            msg = self.socket.recv(flags=zmq.NOBLOCK)
        except zmq.ZMQError :
            msg = None
        if msg:
            confirmation = "Unknown error in receive logic."
            if msg[0] == 'R':
                rec_name = msg[2:]
                for p in self.g_pool.plugins:
                    if p.class_name == 'Recorder':
                        if p.running:
                            confirmation = "Stopped recording."
                        else:
                            if rec_name:
                                self.g_pool.rec_name = rec_name
                            confirmation = 'started recording with name: %s'%self.g_pool.rec_name
                        p.toggle()
                        break
            elif msg == 'T':
                self.g_pool.timebase.value = self.g_pool.capture.get_now()
                confirmation = "New timebase set to %s all timestamps will count from here now."%self.g_pool.timebase.value
                logger.info("New timebase set to %s all timestamps will count from here now."%self.g_pool.timebase.value)
            elif msg == 'C':
                for p in self.g_pool.plugins:
                    if p.base_class_name == 'Calibration_Plugin':
                        try:
                            p.toggle()
                            if p.active:
                                confirmation = 'Started Calibration: "%s"'%p.class_name
                            else:
                                confirmation = 'Stopped Calibration: "%s"'%p.class_name
                        except AttributeError:
                            confirmation = "'%s' does not support remote start."%p.class_name
                        break
            else:
                confirmation = 'Unknown command.'

            self.socket.send(confirmation)


    def get_init_dict(self):
        return {'address':self.address}


    def close(self):
        self.alive = False

    def cleanup(self):
        """gets called when the plugin get terminated.
           This happens either volunatily or forced.
        """
        self.deinit_gui()
        self.context.destroy()

