import zmq
import ujson as json
import logging
import threading
from zmq.utils.monitor import recv_monitor_message
class ZMQ_handler(logging.Handler):
    '''
    A handler that send log records as json strings via zmq
    '''
    def __init__(self,ctx,ipc_pub_url):
        super(ZMQ_handler, self).__init__()
        self.socket = Msg_Dispatcher(ctx,ipc_pub_url)

    def emit(self, record):
        self.socket.send('logging.%s'%str(record.levelname).lower(),record)

class Msg_Receiver(object):
    '''
    Recv messages on a sub port.
    Not threadsave. Make a new one for each thread
    __init__ will block until connection is established.
    '''
    def __init__(self,ctx,url,topics = (),block_unitl_connected=True):
        self.socket = zmq.Socket(ctx,zmq.SUB)
        assert type(topics) != str

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

    def __del__(self):
        self.socket.close()


class Msg_Dispatcher(object):
    '''
    Send messages on a pub port.
    Not threadsave. Make a new one for each thread
    __init__ will block until connection is established.
    '''
    def __init__(self,ctx,url,block_unitl_connected=True):
        self.socket = zmq.Socket(ctx,zmq.PUB)

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

    def send(self,topic,payload):
        '''
        send a generic message with topic, payload
        '''
        self.socket.send(str(topic),flags=zmq.SNDMORE)
        self.socket.send(json.dumps(payload))

    def notify(self,notification):
        '''
        send a pupil notification
        notificaiton is a dict with a least a subject field
        if a 'delay' field exsits the notification it will be grouped with notifications
        of same subject and only one send after specified delay.
        '''
        if notification.get('delay',0):
            self.send("delayed_notify.%s"%notification['subject'],notification)
        else:
            self.send("notify.%s"%notification['subject'],notification)


    def __del__(self):
        self.socket.close()




if __name__ == '__main__':
    #tap into the IPC backbone of pupil capture
    ctx = zmq.Context()

    # the requester talks to Pupil remote and recevied the session unique IPC SUB URL
    requester = ctx.socket(zmq.REQ)
    requester.connect('tcp://localhost:50020')

    requester.send('SUB_URL')
    ipc_sub_url = requester.recv()
    # we then connect to monitor log messages using the url. We can also monitor other topic if we wish
    print ipc_sub_url
    monitor = Msg_Receiver(ctx,ipc_sub_url,topics=('logging','notify')) #more topics: gaze, pupil, notify, ...

    # now lets get the current pupil time.
    requester.send('t')
    print requester.recv()

    # listen to log messages.
    while True:
        monitor.socket.poll()
        print monitor.recv()
