'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
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
        super().__init__(g_pool)
        self.menu = None
        self._name = name
        self._active_group = active_group
        self.group_members = {}
        self.thread_pipe = None
        self.start_group_communication()

    def init_gui(self):
        def close():
            self.alive = False

        help_str = "Pupil Groups utilizes the ZeroMQ Realtime Exchange Protocol to discover other local group members. We use it to relay notifications to other group members. Example: Sychronise time."
        self.menu = ui.Growing_Menu('Pupil Groups')
        self.menu.append(ui.Button('Close',close))
        self.menu.append(ui.Info_Text(help_str))
        self.menu.append(ui.Text_Input('name',self,label='Name:'))
        self.menu.append(ui.Text_Input('active_group',self,label='Group:'))
        self.group_menu = ui.Growing_Menu('Other Group Members')
        self.update_member_menu()
        self.menu.append(self.group_menu)
        self.g_pool.sidebar.append(self.menu)


    def start_group_communication(self):
        if self.thread_pipe:
            self.stop_group_communication()
        logger.debug('Starting Pupil Groups...')
        self.thread_pipe = zhelper.zthread_fork(self.g_pool.zmq_ctx, self._thread_loop)

    def stop_group_communication(self):
        logging.debug('Stopping Pupil Groups...')
        self.thread_pipe.send_string('$TERM')
        while self.thread_pipe:
            time.sleep(.1)
        logger.info('Pupil Groups stopped.')

    def on_notify(self,notification):
        """Local network discovery and many-to-many communication.

        Reacts to notifications:
            ``groups.name_should_change``: Changes node name
            ``groups.active_group_should_change``: Changes active group
            ``groups.member_joined``: Adds peer to member list.
            ``groups.member_left``: Removes peer from member list.
            ``groups.ping``: Answers with ``groups.pong``
            ``groups.pong``: Log ping/pong roundtrip time

        Emits notifications:
            ``groups.member_joined``: New member appeared.
            ``groups.member_left``: A member left. Might occure multiple times.
            ``groups.ping``: Inits roundtrip time measurement
            ``groups.pong``: Answer to ``groups.ping``
        """
        if notification['subject'].startswith('groups.name_should_change'):
            self.name = notification['name']
        elif notification['subject'].startswith('groups.active_group_should_change'):
            self.active_group = notification['name']
        elif notification['subject'].startswith('groups.member_joined'):
            uuid = notification['uuid_bytes']
            self.group_members[uuid] = notification['name']
            self.update_member_menu()
        elif notification['subject'].startswith('groups.member_left'):
            uuid = notification['uuid_bytes']
            try:
                del self.group_members[uuid]
            except KeyError:
                pass # Already removed from list
            else:
                # Update only on change
                self.update_member_menu()
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
            logger.info('{}: Ping time: {} - Pong time: {}'.format(
                peer['name'],
                float(notification['t2']) - float(notification['t1']),
                float(peer['arrival_timestamp']) - float(notification['t2'])))

    def get_init_dict(self):
        return {'name':self.name, 'active_group': self.active_group}

    def cleanup(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None
        self.stop_group_communication()

    def update_member_menu(self):
        if self.menu:
            self.group_menu.elements[:] = []
            if not self.group_members:
                self.group_menu.append(ui.Info_Text('There are no other group members.'))
            for name in self.group_members.values():
                self.group_menu.append(ui.Info_Text(name))

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
        self._name = value
        self.group_members = {}
        self.thread_pipe.send_string('$RESTART')

    @property
    def active_group(self):
        return self._active_group

    @active_group.setter
    def active_group(self,value):
        self._active_group = value
        self.group_members = {}
        self.update_member_menu()
        self.thread_pipe.send_string('$RESTART')


    # @groups.setter
    # def groups(self,group_list):
    #     self._groups

    # ---------------------------------------------------------------
    # Background functions

    def _thread_loop(self,context,pipe):
        # Pyre helper functions
        def setup_group_member():
            group_member = Pyre(self.name)
            # set headers
            for header in self.default_headers:
                group_member.set_header(*header)
            # join active group
            group_member.join(self.active_group)

            # start group_member
            group_member.start()
            return group_member

        def shutdown_group_member(node):
            node.leave(self.active_group)
            node.stop()

        # setup sockets
        local_in  = Msg_Receiver(context, self.g_pool.ipc_sub_url, topics=('remote_notify.',))
        local_out = Msg_Dispatcher(context, self.g_pool.ipc_push_url)
        group_member = setup_group_member()

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
                    group_member.shout(self.active_group, serialized)
                else:
                    peer_uuid_bytes = notification[remote_key]
                    del notification[remote_key]
                    serialized = serializer.dumps(notification)
                    peer_uuid = uuid.UUID(bytes=peer_uuid_bytes)
                    group_member.whisper(peer_uuid, serialized)

            if group_member.socket() in readable:
                event = PyreEvent(group_member)
                if event.msg:
                    for msg in event.msg:
                        try:
                            # try to unpack data
                            notification = serializer.loads(msg, encoding='utf-8')
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
                        except Exception:
                            logger.info('Dropped garbage data by peer {} ({})'.format(event.peer_name, event.peer_uuid))
                elif event.type == 'JOIN' and event.group == self.active_group:
                    local_out.notify({
                        'subject': 'groups.member_joined',
                        'name': event.peer_name,
                        'uuid_bytes': event.peer_uuid_bytes
                    })
                elif (event.type == 'LEAVE' and event.group == self.active_group) or event.type == 'EXIT':
                    local_out.notify({
                        'subject': 'groups.member_left',
                        'name': event.peer_name,
                        'uuid_bytes': event.peer_uuid_bytes})

            if pipe in readable:
                command = pipe.recv_string()
                if command == '$RESTART':
                    # Restart group_member node to change name
                    poller.unregister(group_member.socket())
                    shutdown_group_member(group_member)
                    group_member = setup_group_member()
                    poller.register(group_member.socket(),zmq.POLLIN)
                elif command == '$TERM':
                    break

        del local_in
        del local_out
        shutdown_group_member(group_member)
        self.thread_pipe = None
