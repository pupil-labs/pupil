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
from multiprocessing import cpu_count
import background_helper as bh

import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Make all pupil shared_modules available to this Python session.
    pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
    sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))

from plugin import Analysis_Plugin_Base

from exporter import export
from player_methods import is_pupil_rec_dir


def get_recording_dirs(data_dir):
    '''
        You can supply a data folder or any folder
        - all folders within will be checked for necessary files
        - in order to make a visualization
    '''
    if is_pupil_rec_dir(data_dir):
        yield data_dir
    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            joined = os.path.join(root, d)
            if not d.startswith(".") and is_pupil_rec_dir(joined):
                yield joined


class Batch_Exporter(Analysis_Plugin_Base):
    """docstring for Batch_Exporter
    this plugin can export videos in a seperate process using exporter
    """
    icon_chr = chr(0xec05)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool, source_dir='~/work/pupil/recordings/BATCH', destination_dir='~/'):
        super().__init__(g_pool)

        self.available_exports = []
        self.queued_exports = []
        self.active_exports = []
        self.previous_exports = []
        self.destination_dir = os.path.expanduser(destination_dir)
        self.source_dir = os.path.expanduser(source_dir)

        self.search_task = None
        self.worker_count = cpu_count() - 1
        logger.info("Using a maximum of {} CPUs to process visualizations in parallel...".format(cpu_count() - 1))

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Batch Export Recordings'

        self.menu.append(ui.Info_Text('Search will walk through the source direcotry recursively and detect available Pupil recordings.'))
        self.menu.append(ui.Text_Input('source_dir', self, label='Source Directory', setter=self.set_src_dir))

        self.search_button = ui.Button('Search', self.detect_recordings)
        self.menu.append(self.search_button)

        self.avail_recs_menu = ui.Growing_Menu('Available Recordings')
        self._update_avail_recs_menu()
        self.menu.append(self.avail_recs_menu)

        self.menu.append(ui.Text_Input('destination_dir', self, label='Destination Directory', setter=self.set_dest_dir))
        self.menu.append(ui.Button('Export selected', self.export_selected))
        self.menu.append(ui.Button('Clear search results', self._clear_avail))

        self._update_ui()

    def _update_ui(self):
        del self.menu.elements[7:]

        if self.queued_exports:
            self.menu.append(ui.Separator())
            self.menu.append(ui.Info_Text('Queued exports:'))
            for queued in self.queued_exports[::-1]:
                def prio_qd():
                    pass
                self.menu.append(ui.Button('Prioritize', prio_qd, outer_label=queued['dest']))

        if self.active_exports:
            self.menu.append(ui.Separator())

            for job in self.active_exports[::-1]:
                submenu = ui.Growing_Menu('Active: {}'.format(job.out_file_path))
                submenu.append(ui.Text_Input('status', job, label='Status', setter=lambda x: None))
                progress_bar = ui.Slider('progress', job, min=0, max=job.frames_to_export, label='Progress')
                progress_bar.read_only = True
                submenu.append(progress_bar)
                submenu.append(ui.Button('Cancel', job.cancel))
                self.menu.append(submenu)

        if self.previous_exports:
            self.menu.append(ui.Separator())

            for job in self.previous_exports[::-1]:
                if job.completed:
                    status = 'Completed'
                elif job.canceled:
                    status = 'Canceled'
                else:
                    status = 'Previous'
                submenu = ui.Growing_Menu('{}: {}'.format(status, job.out_file_path))
                progress_bar = ui.Slider('progress', job, min=0, max=job.frames_to_export, label='Progress')
                progress_bar.read_only = True
                submenu.append(progress_bar)
                submenu.collapsed = True
                self.menu.append(submenu)

            self.menu.append(ui.Button('Clear previous exports', self._clear_previous))

    def deinit_ui(self):
        self.menu.remove(self.avail_recs_menu)
        self.avail_recs_menu = None
        self.remove_menu()

    def detect_recordings(self):
        if self.search_task:
            self.search_task.cancel()
            self.search_task = None
            self.search_button.outer_label = ''
            self.search_button.label = 'Search'
        else:
            self.search_button.outer_label = 'Searching...'
            self.search_button.label = 'Cancel'
            self.search_task = bh.Task_Proxy('Searching recordings in {}'.format(self.source_dir), get_recording_dirs, args=[self.source_dir])

    def set_src_dir(self, new_dir):
        new_dir = os.path.expanduser(new_dir)
        if os.path.isdir(new_dir):
            self.source_dir = new_dir
        else:
            logger.warning('"{}" is not a directory'.format(new_dir))
            return

    def set_dest_dir(self, new_dir):
        new_dir = os.path.expanduser(new_dir)
        if os.path.isdir(new_dir):
            self.destination_dir = new_dir
        else:
            logger.warning('"{}" is not a directory'.format(new_dir))
            return

    def _clear_previous(self):
        del self.previous_exports[:]
        self._update_ui()

    def _clear_avail(self):
        del self.available_exports[:]
        self._update_avail_recs_menu()

    def _update_avail_recs_menu(self):
        del self.avail_recs_menu[:]
        if self.available_exports:
            for avail in self.available_exports:
                self.avail_recs_menu.append(ui.Switch('selected', avail, label=avail['source']))
        else:
            self.avail_recs_menu.append(ui.Info_Text('No recordings available yet. Use Search to find recordings.'))

    def export_selected(self):
        for avail in self.available_exports[:]:
            if avail['selected']:
                try:
                    frames_to_export = len(np.load(os.path.join(avail['source'], 'world_timestamps.npy')))
                except:
                    logger.error('Invalid export directory: {}'.format(avail['source']))
                    self.available_exports.remove(avail)
                    continue

                # make a unique name created from rec_session and dir name
                rec_session, rec_dir = avail['source'].rsplit(os.path.sep, 2)[1:]
                out_name = rec_session+"_"+rec_dir+".mp4"
                out_file_path = os.path.join(self.destination_dir, out_name)

                if (out_file_path in (e['dest'] for e in self.queued_exports) or
                        out_file_path in (e.out_file_path for e in self.active_exports)):
                    logger.error("This export setting would try to save {} at least twice please rename dirs to prevent this. Skipping recording.".format(out_file_path))
                    continue
                if out_file_path in (e.out_file_path for e in self.previous_exports):
                    logger.error("This export setting would the previous export {}. Please clear the previous exports if you want to overwrite it.".format(out_file_path))
                    continue

                export = {'source': avail['source'], 'dest': out_file_path, 'frames_to_export': frames_to_export}
                self.available_exports.remove(avail)
                self.queued_exports.append(export)
        self._update_avail_recs_menu()
        self._update_ui()

    def recent_events(self, events):
        if self.search_task:
            recent = [d for d in self.search_task.fetch()]
            if recent:
                currently_avail = [rec['source'] for rec in self.available_exports]
                self.available_exports.extend([{'source': rec, 'selected': True} for rec in recent if rec not in currently_avail])
                self._update_avail_recs_menu()
            if self.search_task.completed:
                self.search_task = None
                self.search_button.outer_label = ''
                self.search_button.label = 'Search'


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
