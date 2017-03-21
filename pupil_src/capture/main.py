'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import os, sys, platform

# sys.argv.append('profiled')
# sys.argv.append('debug')
# sys.argv.append('service')

app = 'capture'

if getattr(sys, 'frozen', False):
    if 'pupil_service' in sys.executable:
        app = 'service'
    # Specifiy user dir.
    user_dir = os.path.expanduser(os.path.join('~', 'pupil_{}_settings'.format(app)))
    version_file = os.path.join(sys._MEIPASS,'_version_string_')
else:
    if 'service' in sys.argv:
        app = 'service'
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))
    # Specifiy user dir.
    user_dir = os.path.join(pupil_base_dir,'{}_settings'.format(app))
    version_file = None

# create folder for user settings, tmp data
if not os.path.isdir(user_dir):
    os.mkdir(user_dir)

# create folder for user plugins
plugin_dir = os.path.join(user_dir,'plugins')
if not os.path.isdir(plugin_dir):
    os.mkdir(plugin_dir)

#app version
from version_utils import get_version
app_version = get_version(version_file)

#threading and processing
from multiprocessing import Process, Value,active_children,freeze_support,set_start_method
from threading import Thread
from ctypes import c_double,c_bool

#networking
import zmq
import zmq_tools

#time
from time import time,sleep

# os utilities
from os_utils import Prevent_Idle_Sleep

#functions to run in seperate processes
if 'profiled' in sys.argv:
    from world import world_profiled as world
    from service import service_profiled as service
    from eye import eye_profiled as eye
else:
    from world import world
    from service import service
    from eye import eye



def launcher():
    """Starts eye processes. Hosts the IPC Backbone and Logging functions.

    Reacts to notifications:
       ``launcher_process.should_stop``: Stops the launcher process
       ``eye_process.should_start``: Starts the eye process
    """


    #Reliable msg dispatch to the IPC via push bridge.
    def pull_pub(ipc_pub_url,pull):
        ctx = zmq.Context.instance()
        pub = ctx.socket(zmq.PUB)
        pub.connect(ipc_pub_url)

        while True:
            m = pull.recv_multipart()
            pub.send_multipart(m)


    #The delay proxy handles delayed notififications.
    def delay_proxy(ipc_pub_url,ipc_sub_url):
        ctx = zmq.Context.instance()
        sub = zmq_tools.Msg_Receiver(ctx,ipc_sub_url,('delayed_notify',))
        pub = zmq_tools.Msg_Dispatcher(ctx,ipc_pub_url)
        poller = zmq.Poller()
        poller.register(sub.socket, zmq.POLLIN)
        waiting_notifications = {}

        while True:
            if poller.poll(timeout=250):
                #Recv new delayed notification and store it.
                topic,n = sub.recv()
                n['_notify_time_'] = time()+n['delay']
                waiting_notifications[n['subject']] = n
            #When a notifications time has come, pop from dict and send it as notification
            for s,n in list(waiting_notifications.items()):
                if n['_notify_time_'] < time():
                    del n['_notify_time_']
                    del n['delay']
                    del waiting_notifications[s]
                    pub.notify(n)


    #Recv log records from other processes.
    def log_loop(ipc_sub_url,log_level_debug):
        import logging
        #Get the root logger
        logger = logging.getLogger()
        #set log level
        if log_level_debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        #Stream to file
        fh = logging.FileHandler(os.path.join(user_dir,'capture.log'),mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(processName)s - [%(levelname)s] %(name)s: %(message)s'))
        logger.addHandler(fh)
        #Stream to console.
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(processName)s - [%(levelname)s] %(name)s: %(message)s'))
        logger.addHandler(ch)
        # IPC setup to receive log messages. Use zmq_tools.ZMQ_handler to send messages to here.
        sub = zmq_tools.Msg_Receiver(zmq_ctx,ipc_sub_url,topics=("logging",))
        while True:
            topic,msg = sub.recv()
            record = logging.makeLogRecord(msg)
            logger.handle(record)


    ## IPC
    timebase = Value(c_double,0)
    eyes_are_alive = Value(c_bool,0),Value(c_bool,0)

    zmq_ctx = zmq.Context()

    #Let the OS choose the IP and PORT
    ipc_pub_url = 'tcp://*:*'
    ipc_sub_url = 'tcp://*:*'
    ipc_push_url = 'tcp://*:*'

    # Binding IPC Backbone Sockets to URLs.
    # They are used in the threads started below.
    # Using them in the main thread is not allowed.
    xsub_socket = zmq_ctx.socket(zmq.XSUB)
    xsub_socket.bind(ipc_pub_url)
    ipc_pub_url = xsub_socket.last_endpoint.decode('utf8').replace("0.0.0.0","127.0.0.1")

    xpub_socket = zmq_ctx.socket(zmq.XPUB)
    xpub_socket.bind(ipc_sub_url)
    ipc_sub_url = xpub_socket.last_endpoint.decode('utf8').replace("0.0.0.0","127.0.0.1")

    pull_socket = zmq_ctx.socket(zmq.PULL)
    pull_socket.bind(ipc_push_url)
    ipc_push_url = pull_socket.last_endpoint.decode('utf8').replace("0.0.0.0","127.0.0.1")


    # Starting communication threads:
    # A ZMQ Proxy Device serves as our IPC Backbone
    ipc_backbone_thread = Thread(target=zmq.proxy, args=(xsub_socket,xpub_socket))
    ipc_backbone_thread.setDaemon(True)
    ipc_backbone_thread.start()

    pull_pub = Thread(target=pull_pub, args=(ipc_pub_url,pull_socket))
    pull_pub.setDaemon(True)
    pull_pub.start()

    log_thread = Thread(target=log_loop, args=(ipc_sub_url,'debug'in sys.argv))
    log_thread.setDaemon(True)
    log_thread.start()

    delay_thread = Thread(target=delay_proxy, args=(ipc_push_url,ipc_sub_url))
    delay_thread.setDaemon(True)
    delay_thread.start()

    del xsub_socket,xpub_socket,pull_socket
    sleep(0.2)

    topics = (  'notify.eye_process.',
                'notify.launcher_process.',
                'notify.meta.should_doc')
    cmd_sub = zmq_tools.Msg_Receiver(zmq_ctx,ipc_sub_url,topics=topics )
    cmd_push = zmq_tools.Msg_Dispatcher(zmq_ctx,ipc_push_url)

    if app == 'service':
        Process(target=service,
                      name= 'service',
                      args=(timebase,
                            eyes_are_alive,
                            ipc_pub_url,
                            ipc_sub_url,
                            ipc_push_url,
                            user_dir,
                            app_version
                            )).start()
    else:
        Process(target=world,
                      name= 'world',
                      args=(timebase,
                            eyes_are_alive,
                            ipc_pub_url,
                            ipc_sub_url,
                            ipc_push_url,
                            user_dir,
                            app_version,
                            )).start()

    with Prevent_Idle_Sleep():
        while True:
            #block and listen for relevant messages.
            topic,n = cmd_sub.recv()
            if "notify.eye_process.should_start" in topic:
                eye_id = n['eye_id']
                if not eyes_are_alive[eye_id].value:
                    Process(target=eye, name='eye{}'.format(eye_id), args=(
                            timebase,
                            eyes_are_alive[eye_id],
                            ipc_pub_url,
                            ipc_sub_url,
                            ipc_push_url,
                            user_dir,
                            app_version,
                            eye_id
                            )).start()
            elif "notify.launcher_process.should_stop" in topic:
                break
            elif "notify.meta.should_doc" in topic:
                cmd_push.notify({
                    'subject':'meta.doc',
                    'actor':'launcher',
                    'doc':launcher.__doc__})

        for p in active_children(): p.join()


if __name__ == '__main__':
    freeze_support()
    if platform.system() == 'Darwin':
        set_start_method('spawn')
    launcher()
