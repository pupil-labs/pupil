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

class Pupil_Groups(Plugin):
    """Interface for local network discovery and many-to-many communication.

    Uses Pyre for local group member discovery.
    """
    def __init__(self, g_pool, name="Unnamed Group Member", active_group="pupil-groups"):
        super(Pupil_Groups, self).__init__(g_pool)
        self._name = name
        self._active_group = active_group
        self.thread_pipe = None
        self.start_group_communication()

    def init_gui(self):
        help_str = "Uses the ZeroMQ Realtime Exchange Protocol to discover other local group members."
        self.menu = ui.Growing_Menu('Pupil Groups')
        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('name',self,label='Name:'))
        self.menu.append(ui.Text_Input('active_group',self,label='Group:'))
        self.menu.append(ui.Button('Ping Other Nodes',self.test))
        self.g_pool.sidebar.append(self.menu)


    def start_group_communication(self):
        if self.thread_pipe:
            self.stop_group_communication()
        logger.debug('Starting Pupil Groups...')
        self.thread_pipe = zhelper.zthread_fork(self.g_pool.zmq_ctx, self._thread_loop)

    def stop_group_communication(self):
        logging.debug('Stopping Pupil Groups...')
        self.thread_pipe.send('$TERM')
        while self.thread_pipe:
            time.sleep(.1)
        logger.info('Pupil Groups stopped.')

    def on_notify(self,notification):
        if notification['subject'].startswith('groups.name_should_change'):
            self.name = notification['name']
        elif notification['subject'].startswith('groups.active_group_should_change'):
            self.active_group = notification['name']
        elif notification['subject'].startswith('groups.ping'):
            peer = notification['groups.peer']
            self.notify_all({
                'subject': 'groups.pong',
                't1': notification['t1'],
                't2': self.g_pool.get_timestamp(),
                'remote_notify': peer['uuid_bytes']
            })
        elif notification['subject'].startswith('groups.pong'):
            peer = notification['groups.peer']
            logger.info(
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
        self.stop_group_communication()

    @property
    def default_headers(self):
        return [
            ('sub_address', self.g_pool.ipc_sub_url),
            ('pub_address', self.g_pool.ipc_pub_url),
            ('app_type', self.g_pool.app)
        ]

    def test(self):
        self.notify_all({
            'subject': 'groups.ping',
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
        local_out = Msg_Dispatcher(context, self.g_pool.ipc_push_url)
        group_member = self._setup_group_member()

        # register sockets for polling
        poller = zmq.Poller()
        poller.register(pipe,zmq.POLLIN)
        poller.register(local_in.socket,zmq.POLLIN)
        poller.register(group_member.socket(),zmq.POLLIN)

        logger.info('Pupil Groups started.')

        # Poll loop
        while True:
            # Wait for next readable item
            readable = dict(poller.poll())

            # shout or whisper marked notifications
            if local_in.socket in readable:
                topic, notification = local_in.recv()
                remote_key = 'remote_notify'
                if notification[remote_key] == 'all':
                    del notification[remote_key]
                    serialized = serializer.dumps(notification)
                    group_member.shout(self.active_group,serialized)
                else:
                    peer_uuid_bytes = notification[remote_key]
                    del notification[remote_key]
                    serialized = serializer.dumps(notification)
                    peer_uuid = uuid.UUID(bytes=peer_uuid_bytes)
                    group_member.whisper(peer_uuid,serialized)

            if group_member.socket() in readable:
                event = PyreEvent(group_member)
                if event.msg:
                    for msg in event.msg:
                        try:
                            # try to unpack data
                            notification = serializer.loads(msg)
                            # test if dictionary and if `subject` key is present
                            notification['subject']
                            # add peer information
                            notification['groups.peer'] = {
                                'uuid_bytes': event.peer_uuid_bytes,
                                'name': event.peer_name,
                                'arrival_timestamp': self.g_pool.get_timestamp(),
                                'type': event.type
                            }
                            local_out.notify(notification)
                        except Exception as e:
                            logger.info('Dropped garbage data by peer %s (%s)'%(event.peer_name, event.peer_uuid))

            if pipe in readable:
                command = pipe.recv()
                if command == '$NAME':
                    # Restart group_member node to change name
                    poller.unregister(group_member.socket())
                    self._shutdown_group_member(group_member)
                    group_member = self._setup_group_member()
                    poller.register(group_member.socket(),zmq.POLLIN)
                elif command == '$GROUP':
                    # Leave old group. Join new group.
                    old_group = pipe.recv()
                    new_group = pipe.recv()
                    group_member.leave(old_group)
                    group_member.join(new_group)
                elif command == '$TERM':
                    break

        del local_in
        del local_out
        self._shutdown_group_member(group_member)
        self.thread_pipe = None


    def _setup_group_member(self,start_group_member=True):
        group_member = Pyre(self.name)
        # set headers
        for header in self.default_headers:
            group_member.set_header(*header)
        # join active group
        group_member.join(self.active_group)

        # start group_member
        if start_group_member:
            group_member.start()
        return group_member

    def _shutdown_group_member(self,node):
        node.leave(self.active_group)
        node.stop()