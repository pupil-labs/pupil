'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import collections
from plugin import System_Plugin_Base
import zmq_tools


class Pupil_Data_Relay(System_Plugin_Base):
    """
    """

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.order = .01
        self.gaze_pub = zmq_tools.Msg_Streamer(self.g_pool.zmq_ctx,self.g_pool.ipc_pub_url)
        self.pupil_sub = zmq_tools.Msg_Receiver(self.g_pool.zmq_ctx,self.g_pool.ipc_sub_url,topics=('pupil',))
        self.recent_pupil_data = collections.deque()
        self.recent_gaze_data = collections.deque()

    def recent_events(self, events):
        while self.pupil_sub.new_data:
            topic, pupil_datum = self.pupil_sub.recv()
            self.recent_pupil_data.append(pupil_datum)
            new_gaze_data = self.g_pool.active_gaze_mapping_plugin.on_pupil_datum(pupil_datum)
            for gaze_datum in new_gaze_data:
                self.gaze_pub.send(gaze_datum)
            self.recent_gaze_data.extend(new_gaze_data)

        events['pupil'] = list(self.recent_pupil_data)
        events['gaze'] = list(self.recent_gaze_data)

        self.recent_pupil_data.clear()
        self.recent_gaze_data.clear()
