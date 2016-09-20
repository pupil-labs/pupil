'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import ndsi
from fake_capture import Fake_Capture

class Network_Device_Manager(object):
    """docstring for Network_Device_Manager"""
    def __init__(self):
        super(Network_Device_Manager, self).__init__()
        self.network = ndsi.Network()
        self.network.start()

    def poll_events(self):
        while self.network.has_events:
            self.network.handle_event()

    def device_list(self):
        self.poll_events()
        return [{
            'name': '%s @ %s'%(s['sensor_name'],s['host_name']),
            'uid' : s['sensor_id']
        }
        for s in self.network.sensors.values()
        if s['sensor_type'] == 'video']

    def is_accessible(self, uid):
        return uid in self.network.sensors

    def network_device_capture(self, uid):
        return Network_Device_Capture(self, uid)

class Network_Device_Capture(object):
    """docstring for Network_Device_Capture"""
    def __init__(self, network, uid):
        super(Network_Device_Capture, self).__init__()
        self.sensor = network.sensor(uid, callbacks=(self.on_notification,))
        self.name = '%s @ %s'%(self.sensor.name, self.sensor.host_name)

        self.fake = Fake_Capture()

    def get_frame_robust(self):
        return self.fake.get_frame_robust()

    def on_notification(self, caller, notification):
        pass

    @property
    def frame_size(self):
        return self.fake.frame_size
    @frame_size.setter
    def frame_size(self,new_size):
        self.fake.frame_size(new_size)

    @property
    def frame_rates(self):
        return self.fake.frame_rates

    @property
    def frame_sizes(self):
        return self.fake.frame_sizes

    @property
    def frame_rate(self):
        return self.fake.frame_rate
    @frame_rate.setter
    def frame_rate(self,new_rate):
        self.fake.frame_rate(new_rate)
