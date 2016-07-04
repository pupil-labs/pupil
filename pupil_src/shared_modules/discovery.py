'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import zmq, time, uuid
from pyre import Pyre, PyreEvent, zhelper
from pyglui import ui
from plugin import Plugin
from zmq_tools import Msg_Dispatcher, Msg_Receiver
import msgpack as serializer

import logging
logger = logging.getLogger(__name__)

class Discovery(Plugin):
    """Plugin for local network discovery

    Uses Pyre for local service discovery.
    """
    def __init__(self, g_pool, name="Unnamed Discovery Node", active_group="pupil-discovery"):
        super(Discovery, self).__init__(g_pool)
        self._name = name
        self._active_group = active_group
        self.thread_pipe = None
        self.start_discovery()

    def init_gui(self):
        help_str = "Uses the ZeroMQ Realtime Exchange Protocol to discover other local network members."
        self.menu = ui.Growing_Menu('Discovery')
        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Button('Test',self.test))
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('name',self,label='Name:'))
        self.menu.append(ui.Text_Input('active_group',self,label='Group:'))
        self.g_pool.sidebar.append(self.menu)


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

    def on_notify(self,notification):
        if notification['subject'].startswith('discovery.name_should_change'):
            self.name = notification['name']
        elif notification['subject'].startswith('discovery.active_group_should_change'):
            self.active_group = notification['name']
        elif notification['subject'].startswith('discovery.ping') and 'discovery.peer' in notification:
            peer = notification['discovery.peer']
            self.notify_all({
                'subject': 'discovery.pong',
                't1': notification['t1'],
                't2': self.g_pool.get_timestamp(),
                'remote_notify': peer['uuid']
            })
        elif notification['subject'].startswith('discovery.pong') and 'discovery.peer' in notification:
            peer = notification['discovery.peer']
            logger.debug(
                '%s took %s seconds to answer.'%(
                    peer['name'],
                    float(notification['t2'])-
                    float(notification['t1']))
            )


    def close(self):
        self.alive = False

    def get_init_dict(self):
        return {'name':self.name, 'active_group': self.active_group}

    def cleanup(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None
        self.stop_discovery()

    @property
    def default_headers(self):
        return [
            ('sub_address', self.g_pool.ipc_sub_url),
            ('pub_address', self.g_pool.ipc_pub_url),
            ('app_type', self.g_pool.app)
        ]

    def test(self):
        self.notify_all({
            'subject': 'discovery.ping',
            't1': self.g_pool.get_timestamp(),
            'remote_notify': 'all'
        })

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self,value):
        self.thread_pipe.send('$NAME')
        self.thread_pipe.send(value)
        self._name = value

    @property
    def active_group(self):
        return self._active_group

    @active_group.setter
    def active_group(self,value):
        self.thread_pipe.send('$GROUP')
        self.thread_pipe.send(self._active_group)
        self.thread_pipe.send(value)
        self._active_group = value


    # @groups.setter
    # def groups(self,group_list):
    #     self._groups

    # ---------------------------------------------------------------
    # Background functions

    def _thread_loop(self,context,pipe):
        # setup sockets
        local_in  = Msg_Receiver(context, self.g_pool.ipc_sub_url, topics=('remote_notify.',))
        local_out = Msg_Dispatcher(context, self.g_pool.ipc_sub_url)
        discovery = self._setup_discovery_node()

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

                remote_key = 'remote_notify'
                if notification[remote_key] == 'all':
                    del notification[remote_key]
                    serialized = serializer.dumps(notification)
                    discovery.shout(self.active_group,serialized)
                else:
                    peer_uuid = notification[remote_key]
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
                                'arrival_timestamp': self.g_pool.get_timestamp(),
                                'type': event.type
                            }
                            local_out.notify(notification)
                        except Exception:
                            logger.info('Dropped garbage data by peer %s (%s)'%(event.peer_name, event.peer_uuid))
                            pass

            if pipe in readable:
                command = pipe.recv()
                if command == '$NAME':
                    # Restart discovery node to change name
                    poller.unregister(discovery.socket())
                    self._shutdown_discovery_node(discovery)
                    discovery = self._setup_discovery_node()
                    poller.register(discovery.socket(),zmq.POLLIN)
                elif command == '$GROUP':
                    # Leave old group. Join new group.
                    old_group = pipe.recv()
                    new_group = pipe.recv()
                    discovery.leave(old_group)
                    discovery.join(new_group)
                elif command == '$TERM':
                    break

        del local_in
        del local_out
        self._shutdown_discovery_node(discovery)
        self.thread_pipe = None


    def _setup_discovery_node(self,start_node=True):
        node = Pyre(self.name)
        # set headers
        for header in self.default_headers:
            node.set_header(*header)
        # join active group
        node.join(self.active_group)

        # start node
        if start_node:
            node.start()
        return node

    def _shutdown_discovery_node(self,node):
        node.leave(self.active_group)
        node.stop()