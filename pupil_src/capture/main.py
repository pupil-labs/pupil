'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os, sys, platform

# sys.argv.append('profiled')
# sys.argv.append('debug')
# sys.argv.append('service')

app = 'capture'

if getattr(sys, 'frozen', False):
    if sys.executable.endswith('pupil_service'):
        app = 'service'
    # Specifiy user dir.
    user_dir = os.path.expanduser(os.path.join('~','pupil_%s_settings'%app))
    version_file = os.path.join(sys._MEIPASS,'_version_string_')
else:
    if 'service' in sys.argv:
        app = 'service'
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))
    # Specifiy user dir.
    user_dir = os.path.join(pupil_base_dir,'%s_settings'%app)
    version_file = None

# create folder for user settings, tmp data
if not os.path.isdir(user_dir):
    os.mkdir(user_dir)

#app version
from version_utils import get_version
app_version = get_version(version_file)

#threading and processing
from multiprocessing import Process, Queue, Value,active_children, freeze_support
from threading import Thread
from ctypes import c_double,c_bool

#networking
import zmq
import zmq_tools

#time
from time import time

#functions to run in seperate processes
if 'profiled' in sys.argv:
    from world import world_profiled as world
    from service import service_profiled as service
    from eye import eye_profiled as eye
else:
    from world import world
    from service import service
    from eye import eye



# To assign camera by name: put string(s) in list
world_src = ["Pupil Cam1 ID2","Logitech Camera","(046d:081d)","C510","B525", "C525","C615","C920","C930e"]
eye0_src = ["Pupil Cam1 ID0","HD-6000","Integrated Camera","HD USB Camera","USB 2.0 Camera"]
eye1_src = ["Pupil Cam1 ID1","HD-6000","Integrated Camera"]

# to use a pre-recorded video.
# Use a string to specify the path to your video file as demonstrated below
# world_src = "/Users/mkassner/Downloads/000/world.mkv"
# eye0_src = '/Users/mkassner/Downloads/eye0.mkv'
# eye1_src =  '/Users/mkassner/Downloads/eye.avi'

video_sources = {'world':world_src,'eye0':eye0_src,'eye1':eye1_src}

def main():
    """
    Reacts to notifications:
       ``launcher_process.should_stop``: Stops the launcher process
       ``eye_process.should_start``: Starts the eye process
    """
    ## IPC
    #shared values
    timebase = Value(c_double,0)
    eyes_are_alive = Value(c_bool,0),Value(c_bool,0)

    #network backbone setup
    zmq_ctx = zmq.Context()
    # lets get some open ports:
    test_socket = zmq_ctx.socket(zmq.SUB)
    ipc_sub_port = test_socket.bind_to_random_port('tcp://*', min_port=5001, max_port=6000, max_tries=100)
    ipc_pub_port = test_socket.bind_to_random_port('tcp://*', min_port=6001, max_port=7000, max_tries=100)
    ipc_push_port = test_socket.bind_to_random_port('tcp://*', min_port=6001, max_port=7000, max_tries=100)
    test_socket.close(linger=0)

    ipc_pub_url = 'tcp://127.0.0.1:%s'%ipc_pub_port
    ipc_sub_url = 'tcp://127.0.0.1:%s'%ipc_sub_port
    ipc_push_url = 'tcp://127.0.0.1:%s'%ipc_push_port


    #We use a zmq forwarder and the zmq PUBSUB pattern to do all our IPC.
    def ipc_backbone(in_url, out_url):
        ctx = zmq.Context.instance()
        xsub = ctx.socket(zmq.XSUB)
        xsub.bind(in_url)
        xpub = ctx.socket(zmq.XPUB)
        xpub.bind(out_url)
        try:
            zmq.proxy(xsub, xpub)
        except zmq.ContextTerminated:
            xsub.close()
            xpub.close()

    #reliable msg dispatch to the IPC via push bridge
    def pull_pub(in_url,out_url):
        ctx = zmq.Context.instance()
        pull = ctx.socket(zmq.PULL)
        pull.bind(in_url)
        pub = ctx.socket(zmq.PUB)
        pub.connect(out_url)
        while True:
            m = pull.recv_multipart()
            pub.send_multipart(m)

    #The delay proxy handles delayed notififications.
    def delay_proxy(in_url, out_url):
        ctx = zmq.Context.instance()
        sub = zmq_tools.Msg_Receiver(ctx,in_url,('delayed_notify',))
        pub = zmq_tools.Msg_Dispatcher(ctx,out_url)
        poller = zmq.Poller()
        poller.register(sub.socket, zmq.POLLIN)
        waiting_notifications = {}
        try:
            while True:
                if poller.poll(timeout=250):
                    #Recv new delayed notification and store it.
                    topic,n = sub.recv()
                    n['_notify_time_'] = time()+n['delay']
                    waiting_notifications[n['subject']] = n
                #When a notifications time has come, pop from dict and send it as notification
                for n in waiting_notifications.values():
                    if n['_notify_time_'] < time():
                        del n['_notify_time_']
                        del n['delay']
                        del waiting_notifications[n['subject']]
                        pub.notify(n)
        except zmq.ContextTerminated:
            sub.close()
            pub.close()

    #recv log records from other processes.
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

    ipc_backbone_thread = Thread(target=ipc_backbone, args=('tcp://*:%s'%ipc_pub_port, 'tcp://*:%s'%ipc_sub_port))
    ipc_backbone_thread.setDaemon(True)
    ipc_backbone_thread.start()

    log_thread = Thread(target=log_loop, args=(ipc_sub_url,'debug'in sys.argv))
    log_thread.setDaemon(True)
    log_thread.start()

    delay_thread = Thread(target=delay_proxy, args=(ipc_sub_url, ipc_push_url))
    delay_thread.setDaemon(True)
    delay_thread.start()

    pull_pub = Thread(target=pull_pub, args=('tcp://*:%s'%ipc_push_port, ipc_pub_url))
    pull_pub.setDaemon(True)
    pull_pub.start()


    topics = (  'notify.eye_process.',
                'notify.launcher_process.',
                'notify.notification.should_doc')
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
                            app_version,
                            None)).start()
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
                            video_sources['world'] )).start()



    while True:
        #block and listen for relevant messages.
        topic,n = cmd_sub.recv()
        if "notify.eye_process.should_start" in topic:
            eye_id = n['eye_id']
            if not eyes_are_alive[eye_id].value:
                Process(target=eye,
                            name='eye%s'%eye_id,
                            args=(timebase,
                                eyes_are_alive[eye_id],
                                ipc_pub_url,
                                ipc_sub_url,
                                ipc_push_url,
                                user_dir,
                                app_version,
                                eye_id,
                                video_sources['eye%s'%eye_id] )).start()
        elif "notify.launcher_process.should_stop" in topic:
            break
        elif "notify.notification.should_doc" in topic:
            cmd_push.notify({
                'subject':'notification.doc',
                'actor':'main',
                'doc':main.__doc__})

    for p in active_children(): p.join()

if __name__ == '__main__':
    freeze_support()
    main()
