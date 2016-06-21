'''
This file contains convenience classes for communication with
the Pupil IPC Backbone.
'''

import zmq
from zmq.utils.monitor import recv_monitor_message
import ujson as json
import logging
import threading


class ZMQ_handler(logging.Handler):
    '''
    A handler that send log records as json strings via zmq
    '''
    def __init__(self,ctx,ipc_pub_url):
        super(ZMQ_handler, self).__init__()
        self.socket = Msg_Dispatcher(ctx,ipc_pub_url)

    def emit(self, record):
        self.socket.send('logging.%s'%str(record.levelname).lower(),record)

class ZMQ_Actor(object):
    '''
    Abstract class to unify initilization of Msg_Receiver, Msg_Dispatcher and Requester
    '''
    def __init__(self,ctx,url,socket_type,block_unitl_connected=True):
        super(ZMQ_Actor, self).__init__()
        self.socket = zmq.Socket(ctx,socket_type)
        if block_unitl_connected:
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

    def __del__(self):
        self.socket.close()

class Msg_Receiver(ZMQ_Actor):
    '''
    Recv messages on a sub port.
    Not threadsave. Make a new one for each thread
    __init__ will block until connection is established.
    '''
    def __init__(self,ctx,url,topics = (),block_unitl_connected=True):
        assert type(topics) != str
        super(Msg_Receiver, self).__init__(ctx,url,zmq.SUB,block_unitl_connected)
        for t in topics:
            self.subscribe(t)

    def subscribe(self,topic):
        self.socket.set(zmq.SUBSCRIBE, topic)

    def unsubscribe(self,topic):
        self.socket.set(zmq.UNSUBSCRIBE, topic)

    def recv(self,*args,**kwargs):
        '''
        recv a generic message with topic, payload
        '''
        topic = self.socket.recv(*args,**kwargs)
        payload = json.loads(self.socket.recv(*args,**kwargs))
        return topic,payload

    @property
    def new_data(self):
        return self.socket.get(zmq.EVENTS)


class Msg_Streamer(ZMQ_Actor):
    '''
    Send messages on a pub port.
    Not threadsave. Make a new one for each thread
    '''
    def __init__(self,ctx,url,socket_type=zmq.PUB,block_unitl_connected=True):
        super(Msg_Streamer, self).__init__(ctx,url,socket_type,block_unitl_connected)

    def send(self,topic,payload):
        '''
        send a generic message with topic, payload
        '''
        self.socket.send(str(topic),flags=zmq.SNDMORE)
        self.socket.send(json.dumps(payload))

class Msg_Dispatcher(Msg_Streamer):
    '''
    Send messages on a push port.
    Not threadsave. Make a new one for each thread
    '''
<<<<<<< HEAD
    def __init__(self,ctx,url):
        self.socket = zmq.Socket(ctx,zmq.PUSH)
        self.socket.connect(url)

    def send(self,topic,payload):
        '''
        send a generic message with topic, payload
        '''
        self.socket.send(str(topic),flags=zmq.SNDMORE)
        self.socket.send(json.dumps(payload))
=======
    def __init__(self,ctx,url,block_unitl_connected=True):
        super(Msg_Dispatcher, self).__init__(ctx,url,zmq.PUSH,block_unitl_connected)
>>>>>>> 65fae6f... Revision of zmq_tools

    def notify(self,notification):
        '''
        send a pupil notification
        notification is a dict with a least a subject field
        if a 'delay' field exsits the notification it will be grouped with notifications
        of same subject and only one send after specified delay.
        '''
        if notification.get('delay',0):
            self.send("delayed_notify.%s"%notification['subject'],notification)
        else:
            self.send("notify.%s"%notification['subject'],notification)

class Requester(ZMQ_Actor):
    """
    Send commands or notifications to Pupil Remote
    """
    def __init__(self, ctx, url, block_unitl_connected=True):
        super(Requester, self).__init__(ctx,url,zmq.REQ,block_unitl_connected)

    def send_cmd(self,cmd):
        self.socket.send(cmd)
        return self.socket.recv()

    def notify(self,notification):
        topic = 'notify.' + notification['subject']
        payload = serializer.dumps(notification)
        self.socket.send_multipart((topic,payload))
        return self.socket.recv()

if __name__ == '__main__':
    from time import sleep,time
    #tap into the IPC backbone of pupil capture
    ctx = zmq.Context()

    # the requester talks to Pupil remote and recevied the session unique IPC SUB URL
    requester = ctx.socket(zmq.REQ)
    requester.connect('tcp://localhost:50020')

    requester.send('SUB_PORT')
    ipc_sub_port = requester.recv()
    requester.send('PUB_PORT')
    ipc_pub_port = requester.recv()

    print 'ipc_sub_port:',ipc_sub_port
    print 'ipc_pub_port:',ipc_pub_port

    #more topics: gaze, pupil, logging, ...
    log_monitor = Msg_Receiver(ctx,'tcp://localhost:%s'%ipc_sub_port,topics=('logging.',))
    notification_monitor = Msg_Receiver(ctx,'tcp://localhost:%s'%ipc_sub_port,topics=('notify.',))
    # gaze_monitor = Msg_Receiver(ctx,'tcp://localhost:%s'%ipc_sub_port,topics=('gaze.',))

    #you can also publish to the IPC Backbone directly.
    publisher = Msg_Dispatcher(ctx,'tcp://localhost:%s'%ipc_pub_port)

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
            publisher.notify({'subject':'pingback_test','index':x})
            notification_monitor.recv()
            ts.append(time()-t)
        print min(ts), sum(ts)/len(ts) , max(ts)

    roundtrip_latency_pubsub()
    # now lets get the current pupil time.
    requester.send('t')
    print requester.recv()

    # listen to all notifications.
    while True:
        print notification_monitor.recv()

    # # listen to all log messages.
    # while True:
    #     print log_monitor.recv()