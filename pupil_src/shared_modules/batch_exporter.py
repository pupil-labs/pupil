'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import numpy as np
from pyglui import ui
import os
import sys
import time

from ctypes import c_bool, c_int

import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Make all pupil shared_modules available to this Python session.
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))

from plugin import Analysis_Plugin_Base
from video_export_launcher import mp, Export_Process

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
    for root, dirs, files in os.walk(data_dir):
        filtered_recording_dirs += [os.path.join(root, d) for d in dirs
                                    if not d.startswith(".") and is_pupil_rec_dir(os.path.join(root, d))]
    logger.debug("Filtered Recording Dirs: {}".format(filtered_recording_dirs))
    return filtered_recording_dirs


class Batch_Exporter(Analysis_Plugin_Base):
    """docstring for Batch_Exporter
    this plugin can export videos in a seperate process using exporter
    """
    def __init__(self, g_pool):
        super().__init__(g_pool)

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
        self.workers = [None for x in range(mp.cpu_count())]
        logger.info("Using a maximum of {} CPUs to process visualizations in parallel...".format(mp.cpu_count()))

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
        self.menu.append(ui.Button('Close', self.unset_alive))
        self.menu.append(ui.Text_Input('source_dir', self, label='Recording Source Directory', setter=self.set_src_dir))
        self.menu.append(ui.Text_Input('destination_dir', self, label='Recording Destination Directory', setter=self.set_dest_dir))
        self.menu.append(ui.Button('start export', self.start))

        for idx, job in enumerate(self.exports[::-1]):
            submenu = ui.Growing_Menu("Export Job {}: '{}'".format(idx, job.out_file_path))
            progress_bar = ui.Slider('progress', getter=job.status, min=0, max=job.frames_to_export.value)
            progress_bar.read_only = True
            submenu.append(progress_bar)
            submenu.append(ui.Button('cancel', job.cancel))
            self.menu.append(submenu)
        if not self.exports:
            self.menu.append(ui.Info_Text('Please select a Recording Source directory from with to pull all recordings for export.'))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        return {}

    def set_src_dir(self, new_dir):
        new_dir = new_dir
        self.new_exports = []
        self.exports = []
        new_dir = os.path.expanduser(new_dir)
        if os.path.isdir(new_dir):
            self.source_dir = new_dir
            self.new_exports = get_recording_dirs(new_dir)
        else:
            logger.warning('"{}" is not a directory'.format(new_dir))
            return
        if self.new_exports is []:
            logger.warning('"{}" does not contain recordings'.format(new_dir))
            return

        self.add_exports()
        self._update_gui()

    def set_dest_dir(self, new_dir):
        new_dir = new_dir

        new_dir = os.path.expanduser(new_dir)
        if os.path.isdir(new_dir):
            self.destination_dir = new_dir
        else:
            logger.warning('"{}" is not a directory'.format(new_dir))
            return

        self.exports = []
        self.add_exports()
        self._update_gui()

    def add_exports(self):
        outfiles = set()
        for d in self.new_exports:
            logger.debug("Adding new export.")
            should_terminate = mp.Value(c_bool, False)
            frames_to_export = mp.Value(c_int, 0)
            current_frame = mp.Value(c_int, 0)
            start_frame = None
            end_frame = None
            export_dir = d
            user_dir = self.g_pool.user_dir

            # we need to know the timestamps of our exports.
            try:  # 0.4
                frames_to_export.value = len(np.load(os.path.join(export_dir, 'world_timestamps.npy')))
            except:  # <0.4
                frames_to_export.value = len(np.load(os.path.join(export_dir, 'timestamps.npy')))

            # Here we make clones of every plugin that supports it.
            # So it runs in the current config when we lauch the exporter.
            plugins = self.g_pool.plugins.get_initializers()

            # make a unique name created from rec_session and dir name
            rec_session, rec_dir = export_dir.rsplit(os.path.sep, 2)[1:]
            out_name = rec_session+"_"+rec_dir+".mp4"
            out_file_path = os.path.join(self.destination_dir, out_name)
            if out_file_path in outfiles:
                logger.error("This export setting would try to save {} at least twice please rename dirs to prevent this. Skipping File".format(out_file_path))
            else:
                outfiles.add(out_file_path)
                logger.info("Exporting to: {}".format(out_file_path))

                process = Export_Process(target=export, args=(should_terminate, frames_to_export, current_frame,
                                                              export_dir, user_dir, self.g_pool.min_data_confidence,
                                                              start_frame, end_frame, plugins, out_file_path,None))
                self.exports.append(process)

    def start(self):
        self.active_exports = self.exports[:]
        self.run = True

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
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


def main():

    import argparse
    from textwrap import dedent
    from file_methods import Persistent_Dict

    def show_progess(jobs):
        no_jobs = len(jobs)
        width = 80
        full = width/no_jobs
        string = ""
        for j in jobs:
            try:
                p = int(width*j.current_frame.value/float(j.frames_to_export.value*no_jobs))
            except:
                p = 0
            string += '[' + p*"|" + (full-p)*"-" + "]"
        sys.stdout.write("\r"+string)
        sys.stdout.flush()

    """Batch process recordings to produce visualizations
    Using simple_circle as the default visualizations
    Steps:
        - User Supplies: Directory that contains many recording(s) dirs or just one recordings dir
        - We walk the user supplied directory to get all data folders
        - Data is the list we feed to our multiprocessed
        - Error check -- do we have required files in each dir?: world.avi, gaze_positions.npy, timestamps.npy
        - Result: world_viz.avi within each original data folder
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=dedent('''\
            ***************************************************
            Batch process recordings to produce visualizations
            The default visualization will use simple_circle

            Usage Example:
                python batch_exporter.py -d /path/to/folder-with-many-recordings -s ~/Pupil_Player/settings/user_settings -e ~/my_export_dir
            Arguments:
                -d : Specify a recording directory.
                     This could have one or many recordings contained within it.
                     We will recurse into the dir.
                -s : Specify path to Pupil Player user_settings file to use last used vizualization settings.
                -e : Specify export directory if you dont want the export saved within each recording dir.
                -p : Export a 120 frame preview only.
            ***************************************************\
        '''))
    parser.add_argument('-d', '--rec-dir', required=True)
    parser.add_argument('-s', '--settings-file', required=True)
    parser.add_argument('-e', '--export-to-dir', default=False)
    parser.add_argument('-c', '--basic-color', default='red')
    parser.add_argument('-p', '--preview', action='store_true')

    if len(sys.argv) == 1:
        print(parser.description)
        return

    args = parser.parse_args()
    # get the top level data folder from terminal argument

    data_dir = args.rec_dir

    if args.settings_file and os.path.isfile(args.settings_file):
        session_settings = Persistent_Dict(os.path.splitext(args.settings_file)[0])
        # these are loaded based on user settings
        plugin_initializers = session_settings.get('loaded_plugins', [])
        session_settings.close()
    else:
        logger.error("Setting file not found or valid")
        return

    if args.export_to_dir:
        export_dir = args.export_to_dir
        if os.path.isdir(export_dir):
            logger.info("Exporting all vids to {}".format(export_dir))
        else:
            logger.error("Exporting dir is not valid {}".format(export_dir))
            return
    else:
        export_dir = None
        logger.info("Exporting into the recording dirs.")

    if args.preview:
        preview = True
        logger.info("Exporting first 120frames only")
    else:
        preview = False

    class Temp(object):
        pass

    recording_dirs = get_recording_dirs(data_dir)
    # start multiprocessing engine
    n_cpu = mp.cpu_count()
    logger.info("Using a maximum of {} CPUs to process visualizations in parallel...".format(n_cpu))

    jobs = []
    outfiles = set()
    for d in recording_dirs:
        j = Temp()
        logger.info("Adding new export: {}".format(d))
        j.should_terminate = mp.Value(c_bool, 0)
        j.frames_to_export = mp.Value(c_int, 0)
        j.current_frame = mp.Value(c_int, 0)
        j.data_dir = d
        j.user_dir = None
        j.start_frame = None
        if preview:
            j.end_frame = 30
        else:
            j.end_frame = None
        j.plugin_initializers = plugin_initializers[:]

        if export_dir:
            # make a unique name created from rec_session and dir name
            rec_session, rec_dir = d.rsplit(os.path.sep, 2)[1:]
            out_name = rec_session+"_"+rec_dir+".mp4"
            j.out_file_path = os.path.join(os.path.expanduser(export_dir), out_name)
            if j.out_file_path in outfiles:
                logger.error("This export setting would try to save {} at least twice pleace rename dirs to prevent this.".format(j.out_file_path))
                return
            outfiles.add(j.out_file_path)
            logger.info("Exporting to: {}".format(j.out_file_path))

        else:
            j.out_file_path = None

        j.args = (j.should_terminate, j.frames_to_export, j.current_frame, j.data_dir, j.user_dir,
                  j.start_frame, j.end_frame, j.plugin_initializers, j.out_file_path,None)
        jobs.append(j)

    todo = jobs[:]
    workers = [Export_Process(target=export, args=todo.pop(0).args) for i in range(min(len(todo), n_cpu))]
    for w in workers:
        w.start()

    working = True
    while working:  # cannot use pool as it does not allow shared memory
        working = False
        for i in range(len(workers)):
            if workers[i].is_alive():
                working = True
            else:
                if todo:
                    workers[i] = Export_Process(target=export, args=todo.pop(0).args)
                    workers[i].start()
                    working = True
        show_progess(jobs)
        time.sleep(.25)
    print('\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    main()
