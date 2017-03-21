'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Plugin
import zmq_tools

class Pupil_Data_Relay(Plugin):
    """
    """

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.order = .01
        self.gaze_pub = zmq_tools.Msg_Streamer(self.g_pool.zmq_ctx,self.g_pool.ipc_pub_url)
        self.pupil_sub = zmq_tools.Msg_Receiver(self.g_pool.zmq_ctx,self.g_pool.ipc_sub_url,topics=('pupil',))


    def recent_events(self,events):
        recent_pupil_data = []
        recent_gaze_data = []

        while self.pupil_sub.new_data:
            t,p = self.pupil_sub.recv()
            recent_pupil_data.append(p)
            new_gaze_data = self.g_pool.active_gaze_mapping_plugin.on_pupil_datum(p)
            for g in new_gaze_data:
                self.gaze_pub.send('gaze',g)
            recent_gaze_data += new_gaze_data

        events['pupil_positions'] = recent_pupil_data
        events['gaze_positions'] = recent_gaze_data


    def get_init_dict(self):
        return {}
