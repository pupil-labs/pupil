'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

from plugin import Analysis_Plugin_Base
import os
import time
import multiprocessing
mp = multiprocessing.get_context("fork")

from pyglui import ui
import logging
logger = logging.getLogger(__name__)
from ctypes import c_bool, c_int
from exporter import export


class Export_Process(mp.Process):
    """small aditions to the process class"""
    def __init__(self, target,args):
        super().__init__(target=target,args=args)
        self.should_terminate,self.frames_to_export,self.current_frame,_,_,_,_,_,_,self.out_file_path,_ = args
    def status(self):
        return self.current_frame.value
    def cancel(self):
        self.should_terminate.value = True


def verify_out_file_path(out_file_path,rec_dir):
    #Out file path verification
    if not out_file_path:
        out_file_path = os.path.join(rec_dir, "world_viz.mp4")
    else:
        file_name = os.path.basename(out_file_path)
        dir_name = os.path.dirname(out_file_path)
        if not dir_name:
            dir_name = rec_dir
        if not file_name:
            file_name = 'world_viz.mp4'
        out_file_path = os.path.expanduser(os.path.join(dir_name, file_name))

    out_file_path = avoid_overwrite(out_file_path)
    if os.path.isfile(out_file_path):
        logger.warning("Video out file already exsists. I will overwrite!")
        os.remove(out_file_path)
    logger.debug("Saving Video to {}".format(out_file_path))

    return out_file_path


def avoid_overwrite(out_file_path):
    if os.path.isfile(out_file_path):
        # append something unique to avoid overwriting
        out_file_path, ext = os.path.splitext(out_file_path)
        out_file_path += str(int(time.time())) + '.mp4'
    return out_file_path


class Video_Export_Launcher(Analysis_Plugin_Base):
    """docstring for Video_Export_Launcher
    this plugin can export the video in a seperate process using exporter
    """
    def __init__(self, g_pool):
        super().__init__(g_pool)
        # initialize empty menu
        self.menu = None
        self.new_export = None
        self.exports = []
        # default_path = verify_out_file_path("world_viz.mp4",rec_dir)
        default_path = "world_viz.mp4"
        self.rec_name = default_path

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Export Video')
        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self._update_gui()

    def unset_alive(self):
        self.alive = False

    def _update_gui(self):
        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',self.unset_alive))
        self.menu.append(ui.Info_Text('Supply export video recording name. The export will be in the recording dir. If you give a path the export will end up there instead.'))
        self.menu.append(ui.Text_Input('rec_name',self,label='export name'))
        self.menu.append(ui.Info_Text('Select your export frame range using the trim marks in the seek bar. This will affect all exporting plugins.'))
        self.menu.append(ui.Text_Input('in_mark',getter=self.g_pool.trim_marks.get_string,setter=self.g_pool.trim_marks.set_string,label='frame range to export'))
        self.menu.append(ui.Info_Text("Press the export button or type 'e' to start the export."))

        for job in self.exports[::-1]:
            submenu = ui.Growing_Menu(job.out_file_path)
            progress_bar = ui.Slider('progress', getter=job.status, min=0, max=job.frames_to_export.value)
            progress_bar.read_only = True
            submenu.append(progress_bar)
            submenu.append(ui.Button('cancel',job.cancel))
            self.menu.append(submenu)

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        return {}

    def on_notify(self,notification):
        if notification['subject'] == "should_export":
            self.add_export(notification['range'],notification['export_dir'])

    def add_export(self,export_range,export_dir):
        export_range = slice(*export_range)
        logger.warning("Adding new video export process.")
        should_terminate = mp.Value(c_bool, False)
        frames_to_export  = mp.Value(c_int, 0)
        current_frame = mp.Value(c_int, 0)

        rec_dir = self.g_pool.rec_dir
        user_dir = self.g_pool.user_dir
        start_frame = export_range.start
        end_frame = export_range.stop+1 # end_frame is exclusive
        frames_to_export.value = end_frame-start_frame

        # Here we make clones of every plugin that supports it.
        # So it runs in the current config when we lauch the exporter.
        plugins = self.g_pool.plugins.get_initializers()

        out_file_path = verify_out_file_path(self.rec_name,export_dir)
        pre_computed = {'gaze_positions':self.g_pool.gaze_positions,
                        'pupil_positions':self.g_pool.pupil_positions,
                        'pupil_data':self.g_pool.pupil_data}
        process = Export_Process(target=export, args=(should_terminate,frames_to_export,current_frame, rec_dir,user_dir,self.g_pool.min_data_confidence,start_frame,end_frame,plugins,out_file_path,pre_computed))
        self.new_export = process
        logger.warning("adding export")

    def launch_export(self, new_export):
        logger.debug("Starting export as new process %s" %new_export)
        new_export.start()
        self.exports.append(new_export)
        self._update_gui()

    def recent_events(self, events):
        if self.new_export:
            self.launch_export(self.new_export)
            self.new_export = None

    def gl_display(self):
        pass

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()
        for e in self.exports:
            e.cancel()
            e.join(1.0)
            if e.is_alive():
                logger.error("Export unresponsive - terminating.")
                e.terminate()
