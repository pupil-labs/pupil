'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)

This file contains convenience classes for communication with
the Pupil IPC Backbone.
'''

import zmq
assert zmq.__version__  > '15.1'
from zmq.utils.monitor import recv_monitor_message
# import ujson as serializer # uncomment for json seialization
import msgpack as serializer
import logging


class ZMQ_handler(logging.Handler):
    '''
    A handler that send log records as serialized strings via zmq
    '''
    def __init__(self,ctx,ipc_pub_url):
        super(ZMQ_handler, self).__init__()
        self.socket = Msg_Dispatcher(ctx,ipc_pub_url)

    def emit(self, record):
        self.socket.send('logging.%s'%str(record.levelname).lower(),record.__dict__)

class Msg_Receiver(object):
    '''
    Recv messages on a sub port.
    Not threadsave. Make a new one for each thread
    __init__ will block until connection is established.
    '''
    def __init__(self,ctx,url,topics = (),block_until_connected=True):
        self.socket = zmq.Socket(ctx,zmq.SUB)
        assert type(topics) != str

        if block_until_connected:
            #connect node and block until a connecetion has been made
            monitor = self.socket.get_monitor_socket()
            self.socket.connect(url)
            while True:
                status =  recv_monitor_message(monitor)
                if status['event'] == zmq.EVENT_CONNECTED:
                    break
                elif status['event'] == zmq.EVENT_CONNECT_DELAYED:
                    pass
                else:
                    raise Exception("ZMQ connection failed")
            self.socket.disable_monitor()
        else:
            self.socket.connect(url)

        for t in topics:
            self.subscribe(t)

    def subscribe(self,topic):
        self.socket.set(zmq.SUBSCRIBE, topic)

    def unsubscribe(self,topic):
        self.socket.set(zmq.UNSUBSCRIBE, topic)

    def recv(self):
        '''Recv a message with topic, payload.

        Any addional message frames will be added as a list
        in the payload dict with key: '__raw_data__' .
        '''
        topic = self.socket.recv()
        payload = serializer.loads(self.socket.recv())
        extra_frames = []
        while self.socket.get(zmq.RCVMORE):
            extra_frames.append(self.socket.recv())
        if extra_frames:
            payload['__raw_data__'] = extra_frames
        return topic,payload

    @property
    def new_data(self):
        return self.socket.get(zmq.EVENTS)

    def __del__(self):
        self.socket.close()

class Msg_Streamer(object):
    '''
    Send messages on fast and efficiat but without garatees.
    Not threadsave. Make a new one for each thread
    '''
    def __init__(self,ctx,url):
        self.socket = zmq.Socket(ctx,zmq.PUB)
        self.socket.connect(url)

    def send(self,topic,payload):
        '''Send a message with topic, payload
`
        if payload has the key '__raw_data__' we pop if of the payload and send its raw contents as extra frames
        everything else need to be serializable
        the contents of the iterable in '__raw_data__' require exposing the pyhton buffer interface.
        '''
        if '__raw_data__' not in payload:
            self.socket.send(str(topic),flags=zmq.SNDMORE)
            self.socket.send(serializer.dumps(payload))
        else:
            extra_frames = payload.pop('__raw_data__')
            assert(isinstance(extra_frames, (list, tuple)))
            self.socket.send(str(topic),flags=zmq.SNDMORE)
            self.socket.send(serializer.dumps(payload),flags=zmq.SNDMORE)
            for frame in extra_frames[:-1]:
                self.socket.send(frame,flags=zmq.SNDMORE,copy=True)
            self.socket.send(extra_frames[-1],copy=True)

    def __del__(self):
        self.socket.close()




class Msg_Dispatcher(Msg_Streamer):
    '''
    Send messages with deliavry garantee.
    Not threadsave. Make a new one for each thread
    '''
    def __init__(self,ctx,url):
        self.socket = zmq.Socket(ctx,zmq.PUSH)
        self.socket.connect(url)


    def notify(self,notification):
        '''Send a pupil notification.
        see plugin.notify_all for documentation on notifications.
        '''
        if notification.get('remote_notify'):
            self.send("remote_notify.%s"%notification['subject'],notification)
        elif notification.get('delay',0):
            self.send("delayed_notify.%s"%notification['subject'],notification)
        else:
            self.send("notify.%s"%notification['subject'],notification)



if __name__ == '__main__':
    from time import sleep,time
    #tap into the IPC backbone of pupil capture
    ctx = zmq.Context()

    # the requester talks to Pupil remote and recevied the session unique IPC SUB URL
    requester = ctx.socket(zmq.REQ)
    requester.connect('tcp://127.0.0.1:50020')

    requester.send('SUB_PORT')
    ipc_sub_port = requester.recv()
    requester.send('PUB_PORT')
    ipc_pub_port = requester.recv()

    print 'ipc_sub_port:',ipc_sub_port
    print 'ipc_pub_port:',ipc_pub_port

    #more topics: gaze, pupil, logging, ...
    log_monitor = Msg_Receiver(ctx,'tcp://127.0.0.1:%s'%ipc_sub_port,topics=('logging.',))
    notification_monitor = Msg_Receiver(ctx,'tcp://127.0.0.1:%s'%ipc_sub_port,topics=('notify.',))
    monitor = Msg_Receiver(ctx,'tcp://127.0.0.1:%s'%ipc_sub_port,topics=('pingback_test.3',))
    # gaze_monitor = Msg_Receiver(ctx,'tcp://localhost:%s'%ipc_sub_port,topics=('gaze.',))

    #you can also publish to the IPC Backbone directly.
    publisher = Msg_Streamer(ctx,'tcp://127.0.0.1:%s'%ipc_pub_port)
    sleep(1)
    def roundtrip_latency_reqrep():
        ts = []
        for x in range(100):
            sleep(0.003)
            t = time()
            requester.send('t')
            requester.recv()
            ts.append(time()-t)
        print min(ts), sum(ts)/len(ts) , max(ts)

    def roundtrip_latency_pubsub():
        ts = []
        for x in range(100):
            sleep(0.003)
            t = time()
            publisher.send('pingback_test.3',{'subject':'pingback_test.3','index':x})
            monitor.recv()
            ts.append(time()-t)
        print min(ts), sum(ts)/len(ts) , max(ts)

    #roundtrip_latency_reqrep()
    #roundtrip_latency_pubsub()

    monitor.subscribe('frame.')
    while True:
        topic,msg = monitor.recv()
        print topic,msg['format']

    # # now lets get the current pupil time.
    # requester.send('t')
    # print requester.recv()
    # requester.send_multipart(('notify.service_process.should_stop',serializer.dumps({'subject':'service_process.should_stop'})))
    # print requester.recv()
    # requester.send_multipart(('notify.meta.should_doc',serializer.dumps({'subject':'meta.should_doc'})))
    # print requester.recv()
    # # listen to all notifications.
    # while True:
    #     topic,msg = notification_monitor.recv()
    #     print '%s: %s' %(msg.get('actor'), msg.get('doc'))

    # # listen to all log messages.
    # while True:
    #     print log_monitor.recv()


