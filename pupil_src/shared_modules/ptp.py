from threading import Thread
from threading import Event
import socket
import time
#logging
import logging
logger = logging.getLogger(__name__)






class PTP_Base(object):
    def __init__(self):
        self.port = 2468
        self.server_socket = None
        self.thread = None
        self.active = Event()

    def get_time(self):
        return time.time()

    def overwrite_time_fn(fn):
        self.get_time = fn


class PTP_Slave(PTP_Base):
    def __init__(self,master_ip):
        super(PTP_Slave, self).__init__()
        self.master_ip = master_ip
        self.sample_count = 100

    def sync_packet(self):
      t1 = self.send("sync_packet")
      t, t2 = self.recv()
      return float(t2) - float(t1)

    def delay_packet(self):
      self.send("delay_packet")
      t4, t3 = self.recv()
      return float(t4) - float(t3)

    def recv(self):
        msg = self.server_socket.recv(4096)
        t = self.get_time()
        return (t, msg)

    def send(self,data):
        self.server_socket.sendall(str(data))
        t = self.get_time()
        return t

    def comm_thread(self):
        offsets = []
        delays = []

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as e:
            logger.error("Error creating socket: " + str(e) + ".")
            self.server_socket.close()
            return -1

        self.server_socket.settimeout(2.0)

        logger.debug("Slave connecting to socket... " + str(self.master_ip) + ":" + str(self.port))
        try:
            self.server_socket.connect((self.master_ip, self.port))
        except socket.error as e:
            logger.error("Error connecting to socket: " + e + ".")
            self.server_socket.close()
            return -1

        #sync process:
        try:
          self.send("sync")
          t, resp = self.recv()
          self.send(str(self.sample_count))
          t, resp = self.recv()
        except socket.error as e:
            logger.error("Could not initialize sync: %s"%e)
            self.server_socket.close()
            return -1
        if(resp == "ready"):
            logger.debug("start sync with %s:%s"%(self.master_ip,self.port))
            time.sleep(0.1) #to allow for server to get ready
            try:
                for i in range(self.sample_count):
                    ms_diff = self.sync_packet()
                    sm_diff = self.delay_packet()

                    offset = (ms_diff - sm_diff)/2;
                    delay = (ms_diff + sm_diff)/2;

                    offsets.append(offset)
                    delays.append(delay)
                    self.send("next")
            except socket.error as e:
                logger.error("Could not finish sync: %s"%e)
                self.server_socket.close()
                return -1

        else:
            logger.error("Did not receive correct response")
            self.server_socket.close()
            return -1

        print "\n\nAVG OFFSET: %sms" % str(sum(offsets) * 1000 / len(offsets)) + "\nAVG DELAY: %sms"% str(sum(delays) * 1000 / len(delays))
        print "\n\nMIN OFFSET: %sms" % str(min(offsets) * 1000) + "\nMIN DELAY: %sms"% str(min(delays) * 1000)
        print "\n\nMAX OFFSET: %sms" % str(max(offsets) * 1000) + "\nMAX DELAY: %sms"% str(max(delays) * 1000)
        print "\nDone!"
        return 0

    def start(self):
        self.thread = Thread(target=self.comm_thread, args=())
        self.thread.start()




class PTP_Source(PTP_Base):
    """docstring for PTP_Source"""
    def __init__(self):
        super(PTP_Source, self).__init__()


    def comm_thread(self):

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as e:
            logger.error("Error creating socket: " + str(e) + ".")
            self.server_socket.close()
            return -1

        self.server_socket.settimeout(2.0)

        logger.debug("Source binding to port: "+str(self.port))
        try:
            self.server_socket.connect(('192.168.0.105', self.port))
        except socket.error as e:
            logger.error("Error connecting to socket: " + e + ".")
            self.server_socket.close()
            return -1

        self.server_socket.settimeout(10.0)

        try:
            while self.active:
                logger.debug("Ready to receive requests on port " + str(self.port) + "...")
                data, addr = self.server_socket.recvfrom(4096)
                logger.debug("Sync request from " + addr[0])
                if("sync" == data):
                    self.server_socket.sendto("ready", addr)
                    num_of_times, addr = self.server_socket.recvfrom(4096)
                    self.server_socket.sendto("ready", addr)
                    for i in range(int(num_of_times)):
                        sync_clock()
                    logger.debug("Sync from " + addr[0] + 'complete.')
                else:
                    logger.debug("Ignoring unknowm init message")
            self.server_socket.close()
            logger.debug("Shutting down comm thread")
            return 0

        except socket.error as e:
            logger.error("Error while handling requests: " + str(e))
            self.server_socket.close()
            return -1


    def start(self):
        self.active.set()
        self.thread = Thread(target=self.comm_thread, args=())
        self.thread.start()

    def sync_clock(self):
      addr = self.sync_packet()
      delay_packet(addr)
      recv()

    def sync_packet(self):
      t2, (t1, addr) = self.recv()
      self.send(t2, addr)
      return addr

    def delay_packet(self,addr):
      self.recv()
      self.send(get_time(), addr)

    def recv(self):
        request = self.server_socket.recvfrom(4096)
        t = get_time()
        return (t, request)

    def send(self,data, addr):
        self.server_socket.sendto(str(data), addr)

    def __del__(self):
        self.active.clear()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # source = PTP_Source()
    # source.start()
    slave = PTP_Slave('192.168.1.107')
    slave.start()
    source = None
    time.sleep(20)
