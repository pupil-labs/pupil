"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import functools
import logging
import socket
import socketserver
import struct
import threading
from random import random
from time import sleep

from uvc import get_time_monotonic

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

"""
A Master/Follower scheme for sychronizing clock in a network.
Accuracy is bounded by network latency and clock jitter.

Time synchonization scheme:
    follower sends time req to master at t0
    master returns with massage of own time stamp (t1)
    Upon receipt by master, node take time t2 then: latency: t2-t0 target time = t1+latency/2
    Do this many times. And be smart about what measurments to take and combine.
    node sets timebase to target time.
    apply offset to local time
"""


class Time_Echo(socketserver.BaseRequestHandler):
    """
    Subclass do not use directly!
    reply to request with timestamp
    """

    def __init__(self, *args, time_fn, **kwargs):
        self.time_fn = time_fn
        super().__init__(*args, **kwargs)

    def handle(self):
        while True:
            # expecting `sync` message
            data = self.request.recv(4)
            if not data:
                break
            if data.decode("utf-8") == "sync":
                self.request.send(struct.pack("<d", self.time_fn()))


class Time_Echo_Server(socketserver.ThreadingTCPServer):
    """
    Subclass do not use directly!
    bind at next open port and listen for time sync requests.
    """

    def __init__(self, *, time_fn, host="", **kwargs):
        handler_class = functools.partial(Time_Echo, time_fn=time_fn)
        super().__init__((host, 0), handler_class, **kwargs)
        self.allow_reuse_address = True
        logger.debug(f"Timer Server ready on port: {self.port}")

    @property
    def host(self) -> str:
        return self.server_address[0]

    @property
    def port(self) -> int:
        return self.server_address[1]

    def __del__(self):
        logger.debug("Server closed")


class Clock_Sync_Master(threading.Thread):
    """
    A class that serves clock info to nodes in a local
    network so the can sync their clocks with the Masters one.
    """

    def __init__(self, time_fn):
        threading.Thread.__init__(self)
        self.server = Time_Echo_Server(time_fn=time_fn)
        self.start()

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()
        self.join()
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
    """
    A class that uses jump and slew to adjust a local clock to a remote clock.
    """

    ms = 1 / 1000.0
    us = ms * ms
    tolerance = 0.1 * ms
    max_slew = 500 * us
    min_jump = 10 * ms
    slew_iterations = int(min_jump / max_slew)
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
        self.sync_jitter = 1000000.0

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
                            logger.debug(f"Time adjusted by {offset / self.ms}ms.")
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
                            logger.debug(f"Time slewed by: {slew_time / self.ms}ms")

                            self.in_sync = not bool(offset)
                            self.offset_remains = not self.in_sync
                            if abs(offset) > 0:
                                sleep(self.slew_interval)
                            else:
                                break
                else:
                    logger.debug("No clock adjustent.")
                    self.in_sync = True
                    self.offset_remains = False
                    # print 'no adjustement'
                sleep(self.interval)
            else:
                logger.debug("Failed to connect. Retrying")
                self.in_sync = False
                sleep(self.retry_interval)

            sleep(random())  # wait for a bit to balance load with other nodes.

    def _get_offset(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.settimeout(1.0)
                server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                server_socket.connect((self.host, self.port))
                times = []
                for request in range(60):
                    t0 = self.get_time()
                    server_socket.send(b"sync")
                    message = server_socket.recv(8)
                    t2 = self.get_time()
                    if message:
                        t1 = struct.unpack("<d", message)[0]
                        times.append((t0, t1, t2))

            times.sort(key=lambda t: t[2] - t[0])
            times = times[: int(len(times) * 0.69)]
            # delays = [t2-t0 for t0, t1, t2 in times]
            offsets = [t0 - (t1 + (t2 - t0) / 2) for t0, t1, t2 in times]
            mean_offset = sum(offsets) / len(offsets)
            offset_jitter = sum(abs(mean_offset - o) for o in offsets) / len(offsets)
            # mean_delay = sum(delays)/len(delays)
            # delay_jitter = sum([abs(mean_delay-o)for o in delays])/len(delays)

            # logger.debug('offset: %s (%s),delay %s(%s)'%(mean_offset/self.ms,offset_jitter/self.ms,mean_delay/self.ms,delay_jitter/self.ms))
            return mean_offset, offset_jitter

        except OSError as e:
            logger.debug(f"{e} for {self.host}:{self.port}")
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
                return f"NOT in sync with {self.host}"
            else:
                return "Synced with {}:{} with  {:.2f}ms jitter".format(
                    self.host, self.port, self.sync_jitter / self.ms
                )
        else:
            return f"Connecting to {self.host}"


if __name__ == "__main__":
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
        return get_time_monotonic() + epoch

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

    slave = Clock_Sync_Follower(
        host,
        port=port,
        interval=10,
        time_fn=get_time,
        jump_fn=jump_time,
        slew_fn=slew_time,
    )
    # slave1 = Clock_Sync_Follower(host,port=port,interval=10,time_fn=get_time,jump_fn=jump_time_dummy,slew_fn=slew_time_dummy)
    # slave2 = Clock_Sync_Follower(host,port=port,interval=10,time_fn=get_time,jump_fn=jump_time_dummy,slew_fn=slew_time_dummy)
    # slave3 = Clock_Sync_Follower(host,port=port,interval=10,time_fn=get_time,jump_fn=jump_time_dummy,slew_fn=slew_time_dummy)
    for x in range(10):
        sleep(4)
        print(slave)
        # print "offset:%f, jitter: %f"%(epoch,slave.sync_jitter)
    print("shutting down")
    slave.stop()
    master.stop()
    print("good bye")
