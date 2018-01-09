'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from plugin import Plugin


class Detector_Dummy(Plugin):
    def __init__(g_pool, *args, **kwargs):
        super().__init__(g_pool)

    def detect(self, frame, *args, **kwargs):
        return None

    def visualize(self):
        pass

    def get_settings(self):
        return {}

    def on_resolution_change(self, *args, **kwargs):
        pass
