'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Plugin
import os,sys,time

# see https://github.com/google/python-subprocess32
if os.name == 'posix' and sys.version_info[0] < 3:
    try:
        import subprocess32 as subprocess
    except ImportError:
        import subprocess
else:
    import subprocess
from subprocess import Popen

import logging
logger = logging.getLogger(__name__)

class Caffeine(Plugin):
    """Prevents OS X from sleeping during recording.

    Feature request: https://github.com/pupil-labs/pupil/issues/414
    """

    def __init__(self, g_pool):
        super(Caffeine, self).__init__(g_pool)
        self.order = .9
        if sys.platform != 'darwin':
            logger.error('This plugin only supports OS X currently.')
            self.alive = False
            return
        self.start_caffeine()

    def on_notify(self,notification):
        """Activates ``caffeinate`` during recordings on OS X

        Reacts to notifications:
            ``recording.should_start``: Starts ``caffeinate``
            ``recording.should_stop``: Stops ``caffeinate``
        """
        pass

    def start_caffeine(self):
        app_pid = os.getpid()
        # -w option stops `caffeinate` in case current app does not
        # terminate plugin correctly
        self.caffeine = Popen(['caffeinate', '-disu', '-w', str(app_pid)])
        logger.info('Started caffeine.')

    def stop_caffeine(self):
        if self.caffeine:
            self.caffeine.terminate()
            self.caffeine = None
            logger.info('Stopped caffeine.')

    def cleanup(self):
        self.stop_caffeine()
