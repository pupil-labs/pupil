'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''
from plugin import Menu_Plugin
import logging
logger = logging.getLogger(__name__)

class Calibration_Plugin(Menu_Plugin):
    '''base class for all calibration routines'''
    uniqueness = 'by_base_class'
    def __init__(self,g_pool):
        super().__init__(g_pool)
        self.g_pool.active_calibration_plugin = self
        self.pupil_confidence_threshold = 0.6
        self.active = False


    def init_ui(self):
        calibration_plugins = [p for p in self.g_pool.plugin_by_name.values() if issubclass(p, Calibration_Plugin)]
        from pyglui import ui

        self.menu_icon.label = 'C'
        self.menu_icon.order = 0.3

        def open_plugin(p):
            self.notify_all({'subject':'start_plugin', 'name':p.__name__})

        #We add the capture selection menu
        self.menu.append(ui.Selector(
                                'capture_manager',
                                setter    = open_plugin,
                                getter    = lambda: self.__class__,
                                selection = calibration_plugins,
                                labels    = [p.__name__.replace('_',' ') for p in calibration_plugins],
                                label     = 'Manager'
                            ))


    def on_notify(self,notification):
        '''Handles calibration notifications

        Reacts to notifications:
           ``calibration.should_start``: Starts the calibration procedure
           ``calibration.should_stop``: Stops the calibration procedure

        Emits notifications:
            ``calibration.started``: Calibration procedure started
            ``calibration.stopped``: Calibration procedure stopped
            ``calibration.failed``: Calibration failed
            ``calibration.successful``: Calibration succeeded

        Args:
            notification (dictionary): Notification dictionary
        '''
        if notification['subject'].startswith('calibration.should_start'):
            if self.active:
                logger.warning('Calibration already running.')
            else:
                self.start()
                self.notify_all({'subject':'calibration.started'})
        elif notification['subject'].startswith('calibration.should_stop'):
            if self.active:
                self.notify_all({'subject':'calibration.stopped'})
                self.stop()
            else:
                logger.warning('Calibration already stopped.')

    def toggle(self,_=None):
        if self.active:
            self.notify_all({'subject':'calibration.should_stop'})
        else:
            self.notify_all({'subject':'calibration.should_start'})

    def start(self):
        raise  NotImplementedError()
        self.notify_all({'subject':'calibration.started'})


    def stop(self):
        raise  NotImplementedError()
        self.notify_all({'subject':'calibration.stopped'})