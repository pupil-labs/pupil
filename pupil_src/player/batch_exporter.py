'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Plugin
import numpy as np
from pyglui import ui
import os,sys, platform
import time

import logging
logger = logging.getLogger(__name__)

from ctypes import c_bool, c_int

from export_launcher import Export_Process,Value,forking_enable,cpu_count


from exporter import export
from player_methods import is_pupil_rec_dir

def get_recording_dirs(data_dir):
    '''
        You can supply a data folder or any folder
        - all folders within will be checked for necessary files
        - in order to make a visualization
    '''
    filtered_recording_dirs = []
    if is_pupil_rec_dir(data_dir):
        filtered_recording_dirs.append(data_dir)
    for root,dirs,files in os.walk(data_dir):
        filtered_recording_dirs += [os.path.join(root,d) for d in dirs if not d.startswith(".") and is_pupil_rec_dir(os.path.join(root,d))]
    logger.debug("Filtered Recording Dirs: %s" %filtered_recording_dirs)
    return filtered_recording_dirs

class Batch_Exporter(Plugin):
    """docstring for Export_Launcher
    this plugin can export videos in a seperate process using exporter
    """
    def __init__(self, g_pool):
        super(Batch_Exporter, self).__init__(g_pool)

        # initialize empty menu
        # and load menu configuration of last session
        self.menu = None

        self.exports = []
        self.new_exports = []
        self.active_exports = []
        default_path = os.path.expanduser('~/')
        self.destination_dir = default_path
        self.source_dir = default_path

        self.run = False
        self.workers = [None for x in range(cpu_count())]
        logger.info("Using a maximum of %s CPUs to process visualizations in parallel..." %cpu_count())

    def unset_alive(self):
        self.alive = False

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Batch Export Recordings')
        # load the configuration of last session
        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self._update_gui()

    def _update_gui(self):
        self.menu.elements[:] = []
        self.menu.append(ui.Button('Close',self.unset_alive))
        self.menu.append(ui.Text_Input('source_dir',self,label='Recording Source Directory',setter=self.set_src_dir))
        self.menu.append(ui.Text_Input('destination_dir',self,label='Recording Destination Directory',setter=self.set_dest_dir))
        self.menu.append(ui.Button('start export',self.start))

        for idx,job  in enumerate(self.exports[::-1]):
            submenu = ui.Growing_Menu("Export Job %s: '%s'"%(idx,job.out_file_path))
            progress_bar = ui.Slider('progress', getter=job.status, min=0, max=job.frames_to_export.value)
            progress_bar.read_only = True
            submenu.append(progress_bar)
            submenu.append(ui.Button('cancel',job.cancel))
            self.menu.append(submenu)
        if not self.exports:
            self.menu.append(ui.Info_Text('Please select a Recording Source directory from with to pull all recordings for export.'))



    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        return {}

    def set_src_dir(self,new_dir):
        new_dir = new_dir
        self.new_exports = []
        self.exports = []
        new_dir = os.path.expanduser(new_dir)
        if os.path.isdir(new_dir):
            self.source_dir = new_dir
            self.new_exports = get_recording_dirs(new_dir)
        else:
            logger.warning('"%s" is not a directory'%new_dir)
            return
        if self.new_exports is []:
            logger.warning('"%s" does not contain recordings'%new_dir)
            return

        self.add_exports()
        self._update_gui()

    def set_dest_dir(self,new_dir):
        new_dir = new_dir

        new_dir = os.path.expanduser(new_dir)
        if os.path.isdir(new_dir):
            self.destination_dir = new_dir
        else:
            logger.warning('"%s" is not a directory'%new_dir)
            return

        self.exports = []
        self.add_exports()
        self._update_gui()

    def add_exports(self):
        # on MacOS we will not use os.fork, elsewhere this does nothing.
        forking_enable(0)

        outfiles = set()
        for d in self.new_exports:
            logger.debug("Adding new export.")
            should_terminate = Value(c_bool,False)
            frames_to_export  = Value(c_int,0)
            current_frame = Value(c_int,0)
            start_frame = None
            end_frame = None
            export_dir = d
            user_dir = self.g_pool.user_dir

            #we need to know the timestamps of our exports.
            try: # 0.4
                frames_to_export.value = len(np.load(os.path.join(export_dir,'world_timestamps.npy')))
            except: # <0.4
                frames_to_export.value = len(np.load(os.path.join(export_dir,'timestamps.npy')))

            # Here we make clones of every plugin that supports it.
            # So it runs in the current config when we lauch the exporter.
            plugins = self.g_pool.plugins.get_initializers()


            #make a unique name created from rec_session and dir name
            rec_session, rec_dir = export_dir.rsplit(os.path.sep,2)[1:]
            out_name = rec_session+"_"+rec_dir+".mp4"
            out_file_path = os.path.join(self.destination_dir,out_name)
            if out_file_path in outfiles:
                logger.error("This export setting would try to save %s at least twice please rename dirs to prevent this. Skipping File"%out_file_path)
            else:
                outfiles.add(out_file_path)
                logger.info("Exporting to: %s"%out_file_path)

                process = Export_Process(target=export, args=(should_terminate,frames_to_export,current_frame, export_dir,user_dir,start_frame,end_frame,plugins,out_file_path))
                self.exports.append(process)

    def start(self):
        self.active_exports = self.exports[:]
        self.run = True

    def update(self,frame,events):
        if self.run:
            for i in range(len(self.workers)):
                if self.workers[i] and self.workers[i].is_alive():
                    pass
                else:
                    logger.info("starting new job")
                    if self.active_exports:
                        self.workers[i] = self.active_exports.pop(0)
                        self.workers[i].start()
                    else:
                        self.run = False

    def gl_display(self):
        pass

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happends either voluntary or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self.deinit_gui()


