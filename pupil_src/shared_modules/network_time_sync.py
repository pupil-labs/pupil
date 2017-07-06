'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
from time import sleep
from uvc import get_time_monotonic
import socket
import threading
import asyncore
import struct
from random import random

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

'''
A Master/Follower scheme for sychronizing clock in a network.
Accuracy is bounded by network latency and clock jitter.

Time synchonization scheme:
    follower sends time req to master at t0
    master returns with massage of own time stamp (t1)
    Upon receipt by master, node take time t2 then: latency: t2-t0 target time = t1+latency/2
    Do this many times. And be smart about what measurments to take and combine.
    node sets timebase to target time.
    apply offset to local time
'''


class Time_Echo(asyncore.dispatcher_with_send):
    '''
    Subclass do not use directly!
    reply to request with timestamp
    '''

    def __init__(self, sock, time_fn):
        self.time_fn = time_fn
        asyncore.dispatcher_with_send.__init__(self, sock)

    def handle_read(self):
        # expecting `sync` message
        data = self.recv(4)
        if data:
            self.send(struct.pack('<d', self.time_fn()))

    def __del__(self):
        pass
        # print 'goodbye'


class Time_Echo_Server(asyncore.dispatcher):
    '''
    Subclass do not use directly!
    bind at next open port and listen for time sync requests.
    '''

    def __init__(self, time_fn, socket_map, host=""):
        asyncore.dispatcher.__init__(self, socket_map)
        self.time_fn = time_fn
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind((host, 0))
        self.port = self.socket.getsockname()[1]
        self.listen(5)
        logger.debug('Timer Server ready on port: {}'.format(self.port))

    def handle_accept(self):
        pair = self.accept()
        if pair is not None:
            sock, addr = pair
            logger.debug("syching with %s"%str(addr))
            Time_Echo(sock, self.time_fn)

    def __del__(self):
        logger.debug("Server closed")


class Clock_Sync_Master(threading.Thread):
    '''
    A class that serves clock info to nodes in a local
    network so the can sync their clocks with the Masters one.
    '''
    def __init__(self, time_fn):
        threading.Thread.__init__(self)
        self.socket_map = {}
        self.server = Time_Echo_Server(time_fn, self.socket_map)
        self.start()

    def run(self):
        asyncore.loop(use_poll=True, timeout=1)

    def stop(self):
        # we dont use server.close() as this raises a bad file decritoor exception in loop
        self.server.connected = False
        self.server.accepting = False
        self.server.del_channel()
        self.join()
        self.server.socket.close()
        logger.debug("Server Thread closed")

    def terminate(self):
        self.stop()

    @property
    def port(self):
        return self.server.port

    @property
    def host(self):
        return self.server.host

    def __str__(self):
        return "Acting as clock master."


class Clock_Sync_Follower(threading.Thread):
    '''
    A class that uses jump and slew to adjust a local clock to a remote clock.
    '''
    ms = 1/1000.
    us = ms*ms
    tolerance = 0.1*ms
    max_slew = 500*us
    min_jump = 10*ms
    slew_iterations = int(min_jump/max_slew)
    retry_interval = 1.0
    slew_interval = 0.1

    def __init__(self, host, port, interval, time_fn, jump_fn, slew_fn):
        threading.Thread.__init__(self)
        self.setDaemon(1)
        self.host = host
        self.port = port
        self.interval = interval
        self.running = True
        self.get_time = time_fn
        self.jump_time = jump_fn
        self.slew_time = slew_fn

        # the avg variance of time probes from the last sample run (offset jitter)
        # this error can come from application_runtime jitter, network_jitter,master_clock_jitter and slave_clock_jitter
        self.sync_jitter = 1000000.

        # slave was not able to set the clock at current
        self.offset_remains = True

        # is this node synced?
        self.in_sync = False

        self.start()

    def run(self):
        while self.running:
            result = self._get_offset()
            if result:
                offset, jitter = result
                self.sync_jitter = jitter
                if abs(offset) > max(jitter, self.tolerance):
                    if abs(offset) > self.min_jump:
                        if self.jump_time(offset):
                            self.in_sync = True
                            self.offset_remains = False
                            logger.debug('Time adjusted by {}ms.'.format(offset/self.ms))
                        else:
                            self.in_sync = True
                            self.offset_remains = True
                            sleep(self.retry_interval)
                            continue
                    else:
                        # print 'time slewed required  %sms.'%(offset/self.ms)
                        for x in range(self.slew_iterations):
                            slew_time = max(-self.max_slew, min(self.max_slew, offset))
                            # print offset/self.ms,slew_time/self.ms
                            self.slew_time(slew_time)
                            offset -= slew_time
                            logger.debug('Time slewed by: {}ms'.format(slew_time/self.ms))

                            self.in_sync = not bool(offset)
                            self.offset_remains = not self.in_sync
                            if abs(offset) > 0:
                                sleep(self.slew_interval)
                            else:
                                break
                else:
                    logger.debug('No clock adjustent.')
                    self.in_sync = True
                    self.offset_remains = False
                    # print 'no adjustement'
                sleep(self.interval)
            else:
                logger.debug('Failed to connect. Retrying')
                self.in_sync = False
                sleep(self.retry_interval)

            sleep(random())  # wait for a bit to balance load with other nodes.

    def _get_offset(self):
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.settimeout(1.)
            server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            server_socket.connect((self.host, self.port))
            times = []
            for request in range(60):
                t0 = self.get_time()
                server_socket.send(b'sync')
                message = server_socket.recv(8)
                t2 = self.get_time()
                t1 = struct.unpack('<d', message)[0]
                times.append((t0, t1, t2))

            server_socket.close()

            times.sort(key=lambda t: t[2]-t[0])
            times = times[:int(len(times)*0.69)]
            # delays = [t2-t0 for t0, t1, t2 in times]
            offsets = [t0-((t1+(t2-t0)/2)) for t0, t1, t2 in times]
            mean_offset = sum(offsets)/len(offsets)
            offset_jitter = sum([abs(mean_offset-o)for o in offsets])/len(offsets)
            # mean_delay = sum(delays)/len(delays)
            # delay_jitter = sum([abs(mean_delay-o)for o in delays])/len(delays)

            # logger.debug('offset: %s (%s),delay %s(%s)'%(mean_offset/self.ms,offset_jitter/self.ms,mean_delay/self.ms,delay_jitter/self.ms))
            return mean_offset, offset_jitter

        except socket.error as e:
            logger.debug('{} for {}:{}'.format(e, self.host, self.port))
            return None
        # except Exception as e:
        #     logger.error(str(e))
        #     return 0,0

    def stop(self):
        self.running = False
        self.join()

    def terminate(self):
        self.running = False

    def __str__(self):
        if self.in_sync:
            if self.offset_remains:
                return "NOT in sync with {}".format(self.host)
            else:
                return 'Synced with {}:{} with  {:.2f}ms jitter'.format(self.host,self.port,self.sync_jitter/self.ms)
        else:
            return "Connecting to {}".format(self.host)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    from uvc import get_time_monotonic
    # from time import time as get_time_monotonic
    #### A Note on system clock jitter
    # during tests using a Mac and Linux machine on a 3ms latency network with network jitter of ~50us
    # it became apparent that even on Linux not all clocks are created equal:
    # on MacOS time.time appears to have low jitter (<1ms)
    # on Linux (Ubunut Python 2.7) time.time shows more jitter (<3ms)
    # it is thus recommended for Linux to use uvc.get_time_monotonic.
    master = Clock_Sync_Master(get_time_monotonic)
    port = master.port
    host = "127.0.0.1"
    epoch = 0.0
    # sleep(3)
    # master.stop()

    def get_time():
        return get_time_monotonic()+epoch

    def jump_time(offset):
        global epoch
        epoch -= offset
        return True

    def slew_time(offset):
        global epoch
        epoch -= offset

    def jump_time_dummy(offset):
        return True

    def slew_time_dummy(offset):
        pass

    slave = Clock_Sync_Follower(host,port=port,interval=10,time_fn=get_time,jump_fn=jump_time,slew_fn=slew_time)
    # slave1 = Clock_Sync_Follower(host,port=port,interval=10,time_fn=get_time,jump_fn=jump_time_dummy,slew_fn=slew_time_dummy)
    # slave2 = Clock_Sync_Follower(host,port=port,interval=10,time_fn=get_time,jump_fn=jump_time_dummy,slew_fn=slew_time_dummy)
    # slave3 = Clock_Sync_Follower(host,port=port,interval=10,time_fn=get_time,jump_fn=jump_time_dummy,slew_fn=slew_time_dummy)
    for x in range(10):
        sleep(4)
        print(slave)
        # print "offset:%f, jitter: %f"%(epoch,slave.sync_jitter)
    print('shutting down')
    slave.stop()
    master.stop()
    print('good bye')
