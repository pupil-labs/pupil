"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import zmq_tools
from plugin import System_Plugin_Base


class Pupil_Data_Relay(System_Plugin_Base):
    """"""

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.order = 0.01
        self.gaze_pub = zmq_tools.Msg_Streamer(
            self.g_pool.zmq_ctx, self.g_pool.ipc_pub_url
        )
        self.pupil_sub = zmq_tools.Msg_Receiver(
            self.g_pool.zmq_ctx, self.g_pool.ipc_sub_url, topics=("pupil",)
        )

    def recent_events(self, events):
        recent_pupil_data = []
        recent_gaze_data = []
        while self.pupil_sub.new_data:
            topic, pupil_datum = self.pupil_sub.recv()
            recent_pupil_data.append(pupil_datum)

            gazer = self.g_pool.active_gaze_mapping_plugin
            if gazer is None:
                continue
            for gaze_datum in gazer.map_pupil_to_gaze([pupil_datum]):
                self.gaze_pub.send(gaze_datum)
                recent_gaze_data.append(gaze_datum)

        events["pupil"] = recent_pupil_data
        events["gaze"] = recent_gaze_data
