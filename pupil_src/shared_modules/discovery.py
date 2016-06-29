'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import zmq, time
from pyre import Pyre, PyreEvent, zhelper
from plugin import Plugin
from zmq_tools import Msg_Dispatcher, Msg_Receiver
import msgpack as serializer

import logging
logger = logging.getLogger(__name__)

class Discovery(Plugin):
    """Plugin for local network discovery

    Uses Pyre for local service discovery.
    """
    def __init__(self, g_pool, name=None, groups=["pupil-discovery"]):
        super(Discovery, self).__init__(g_pool)
        self._name = name
        self._groups = groups
        self.thread_pipe = None
        self.start_discovery()


    def start_discovery(self):
        if self.thread_pipe:
            self.stop_discovery()
        logger.debug('Starting discovery...')
        self.thread_pipe = zhelper.zthread_fork(self.g_pool.zmq_ctx, self._thread_loop)

    def stop_discovery(self):
        logging.debug('Stopping discovery...')
        self.thread_pipe.send('$TERM')
        while self.thread_pipe:
            time.sleep(.1)
        logger.info('Discovery stopped.')

    def get_init_dict(self):
        return {}

    def cleanup(self):
        self.stop_discovery()

    @property
    def default_headers(self):
        # TODO: look up request port
        return [
            ('sub_address', self.g_pool.ipc_sub_url),
            ('pub_address', self.g_pool.ipc_pub_url),
            ('app_type', self.g_pool.app)
        ]

    @property
    def name(self):
        return self._name

    @property
    def groups(self):
        return self._groups

    # @groups.setter
    # def groups(self,group_list):
    #     self._groups

    # ---------------------------------------------------------------
    # Background functions

    def _thread_loop(self,context,pipe):
        # setup sockets
        local_in  = Msg_Receiver(context, self.g_pool.ipc_sub_url, topics=('notify.',))
        local_out = Msg_Dispatcher(context, self.g_pool.ipc_sub_url)
        discovery = self._setup_discovery_node()
        discovery.start()

        # register sockets for polling
        poller = zmq.Poller()
        poller.register(pipe,zmq.POLLIN)
        poller.register(local_in.socket,zmq.POLLIN)
        poller.register(discovery.socket(),zmq.POLLIN)

        logger.info('Discovery started.')

        # Poll loop
        while True:
            # Wait for next readable item
            readable = dict(poller.poll())

            # shout or whisper marked notifications
            if local_in in readable:
                topic, notification = local_in.recv()

                # check if notification should be shouted
                shout_key = 'discovery.shout'
                if shout_key in notification:
                    group = notification[shout_key]
                    del notification[shout_key]
                    serialized = serializer.dumps(notification)
                    discovery.shout(group,serialized)

                # check if notification should be whispered
                whisper_key = 'discovery.whisper'
                if whisper_key in notification:
                    peer_uuid = notification[whisper_key]
                    del notification[whisper_key]
                    serialized = serializer.dumps(notification)
                    discovery.whisper(peer_uuid,serialized)

            if discovery.socket() in readable:
                event = PyreEvent(discovery)
                if event.msg:
                    for msg in event.msg:
                        try:
                            # try to unpack data
                            notification = serializer.loads(msg)
                            # test if dictionary and if `subject` key is present
                            notification['subject']
                            # add peer information
                            notification['discovery.peer'] = {
                                'uuid': event.peer_uuid,
                                'name': event.peer_name,
                                'arrival_timestamp': time.time(),
                                'type': event.type
                            }
                            # add group name if notification was shouted
                            if event.type == 'SHOUT':
                                notification['discovery.peer'].update({'group': event.group})
                            local_out.notify(notification)
                        except Exception:
                            logger.info('Dropped garbage data by peer %s (%s)'%(event.peer_name, event.peer_uuid))
                            pass

            if pipe in readable:
                command = pipe.recv()
                if command == '$TERM':
                    break

        del local_in
        del local_out
        self._shutdown_discovery_node(discovery)
        self.thread_pipe = None


    def _setup_discovery_node(self):
        node = Pyre(self.name)
        # set headers
        for header in self.default_headers:
            node.set_header(*header)
        # join groups
        for group in self.groups:
            node.join(group)
        return node

    def _shutdown_discovery_node(self,node):
        for group in self.groups:
            node.leave(group)
        node.stop()