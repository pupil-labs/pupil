import zmq
import ujson as json
import logging
import threading

class ZMQ_handler(logging.Handler):
    '''
    A handler that send log records as json strings via zmq
    '''
    def __init__(self,ctx,ipc_pub_url):
        super(ZMQ_handler, self).__init__()
        self.socket = ctx.socket(zmq.PUB)
        self.socket.connect(ipc_pub_url)

    def emit(self, record):
        self.socket.send_multipart(('logging.%s'%record.levelname.lower(),json.dumps(record)))

# class Msg_Collector(threading.Thread):
#     '''
#     Collect messages on a sub port in a seperate thread.
#     Do not use this for high volume traffic. (GIL battle)
#     Clean exiting not implmentented
#     '''
#     def __init__(self,ctx,url,topics=() ):
#         super(Msg_Collector, self).__init__()
#         assert type(topics) != str
#         self.setDaemon(True)
#         self._ctx = ctx
#         self._url = url
#         self._topics = topics
#         self._new_messages = []

#     def run(self):
#         socket = zmq.backend.Socket(self._ctx,zmq.SUB)
#         socket.connect(self._url)
#         for t in self._topics:
#             socket.set(zmq.SUBSCRIBE, t)

#         while True:
#             topic = socket.recv()
#             msg = json.loads(socket.recv())
#             self._new_messages.append(msg) #append is atomic

#     def collect(self):
#         messages = []
#         while self._new_messages:
#             #worst case, a message is now. We will collect it next time.
#             messages.append(self._new_messages.pop(0)) #pop is atomic
#         return messages

#     def collect_one(self):
#         if self._new_messages:
#             #worst case, a message is added now. We will collect it next time.
#             return self._new_messages.pop(0) #pop is atomic
#         else:
#             return None

class Msg_Receiver(object):
    '''
    Recv messages on a sub port.
    Not threadsave. Make a new one for each thread
    '''
    def __init__(self,ctx,url,topics = ()):
        self.socket = zmq.Socket(ctx,zmq.SUB)
        assert type(topics) != str
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
    '''
    def __init__(self,ctx,url):
        self.socket = zmq.Socket(ctx,zmq.PUB)
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
    sub_url = 'tcp://localhost:5851' #set proper url!
    pub_url = 'tcp://localhost:6851' #set proper url!

    ctx = zmq.Context()
    monitor = Msg_Receiver(ctx,sub_url,topics=('logging','notify')) #more topics: gaze, pupil
    while True:
        print monitor.recv()