'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2018 Pupil Labs

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

from plugin import Analysis_Plugin_Base, System_Plugin_Base

from exporter import export as export_function
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


class Batch_Export(System_Plugin_Base):
    """Sub plugin that manages a single batch export"""
    uniqueness = 'not_unique'
    icon_font = 'pupil_icons'
    icon_chr = chr(0xe2c4)  # character shown in menu icon

    def __init__(self, g_pool, rec_dir, out_file_path, frames_to_export):
        super().__init__(g_pool)
        self.rec_dir = rec_dir
        self.out_file_path = out_file_path
        self.plugins = self.g_pool.plugins.get_initializers()
        self.process = None
        self.status = 'In queue'
        self.progress = 0
        self.frames_to_export = frames_to_export
        self.in_queue = True
        self._accelerate = True
        self.notify_all({'subject': 'batch_export.queued', 'out_file_path': self.out_file_path})

    def init_ui(self):
        self.add_menu()
        # uniqueness = 'not_unique' -> Automatic `Close` button
        # -> Rename to `Cancel`
        export_name = os.path.split(self.out_file_path)[-1]
        self.menu.label = 'Batch Export {}'.format(export_name)
        self.menu[0].label = 'Cancel'
        self.menu_icon.indicator_start = 0.
        self.menu_icon.indicator_stop = 0.1
        self.menu_icon.tooltip = export_name
        self.menu.append(ui.Text_Input('rec_dir', self, label='Recording', setter=lambda x: None))
        self.menu.append(ui.Text_Input('out_file_path', self, label='Output', setter=lambda x: None))
        self.menu.append(ui.Text_Input('status', self, label='Status', setter=lambda x: None))
        progress_bar = ui.Slider('progress', self, min=0, max=self.frames_to_export, label='Progress')
        progress_bar.read_only = True
        self.menu.append(progress_bar)

    def recent_events(self, events):
        if self.process:
            try:
                recent = [d for d in self.process.fetch()]
            except Exception as e:
                self.status, self.progress = '{}: {}'.format(type(e).__name__, e), 0
            else:
                if recent:
                    self.status, self.progress = recent[-1]


            # Update status if process has been canceled or completed
            if self.process.canceled:
                self.process = None
                self.status = 'Export has been canceled.'
                self.notify_all({'subject': 'batch_export.canceled', 'out_file_path': self.out_file_path})
                self.menu[0].label = 'Close'  # change button label back to close
            elif self.process.completed:
                self.process = None
                self.notify_all({'subject': 'batch_export.completed', 'out_file_path': self.out_file_path})
                self.menu[0].label = 'Close'  # change button label back to close

        if self.in_queue:
            if self._accelerate:
                self.menu_icon.indicator_start += 0.01
                self.menu_icon.indicator_stop += 0.02
            else:
                self.menu_icon.indicator_start += 0.02
                self.menu_icon.indicator_stop += 0.01
            d = abs(self.menu_icon.indicator_start - self.menu_icon.indicator_stop)
            if self._accelerate and d > .5:
                self._accelerate = False
            elif not self._accelerate and d < .1:
                self._accelerate = True
        else:
            self.menu_icon.indicator_start = 0.
            self.menu_icon.indicator_stop = self.progress / self.frames_to_export

    def on_notify(self, n):
        if n['subject'] == 'batch_export.should_start' and n['out_file_path'] == self.out_file_path:
            self.init_export()
        if n['subject'] == 'batch_export.should_cancel' and n.get('out_file_path', self.out_file_path) == self.out_file_path:
            self.cancel_export()
            if n.get('remove_menu', False):
                self.alive = False

    def init_export(self):
        self.in_queue = False
        args = (self.rec_dir, self.g_pool.user_dir, self.g_pool.min_data_confidence,
                None, None, self.plugins, self.out_file_path, {})
        self.process = bh.Task_Proxy('Pupil Batch Export {}'.format(self.out_file_path), export_function, args=args)
        self.notify_all({'subject': 'batch_export.started', 'out_file_path': self.out_file_path})

    def cancel_export(self):
        if self.process:
            self.process.cancel()
        self.notify_all({'subject': 'batch_export.canceled', 'out_file_path': self.out_file_path})

    def get_init_dict(self):
        # do not be session persistent
        raise NotImplementedError()

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        self.cancel_export()
        self.notify_all({'subject': 'batch_export.removed', 'out_file_path': self.out_file_path})


class Batch_Exporter(Analysis_Plugin_Base):
    """The Batch_Exporter searches for available recordings and exports them to a common location"""
    icon_chr = chr(0xec05)
    icon_font = 'pupil_icons'

    def __init__(self, g_pool, source_dir='~/', destination_dir='~/'):
        super().__init__(g_pool)

        self.available_exports = []
        self.queued_exports = []
        self.active_exports = []
        self.destination_dir = os.path.expanduser(destination_dir)
        self.source_dir = os.path.expanduser(source_dir)

        self.search_task = None
        self.worker_count = cpu_count() - 1
        logger.info("Using a maximum of {} CPUs to process visualizations in parallel...".format(cpu_count() - 1))

    def get_init_dict(self):
        return {'source_dir': self.source_dir, 'destination_dir': self.destination_dir}

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Batch Export Recordings'

        self.menu.append(ui.Info_Text('Search will walk through the source directory recursively and detect available Pupil recordings.'))
        self.menu.append(ui.Text_Input('source_dir', self, label='Source directory', setter=self.set_src_dir))

        self.search_button = ui.Button('Search', self.detect_recordings)
        self.menu.append(self.search_button)

        self.avail_recs_menu = ui.Growing_Menu('Available Recordings')
        self._update_avail_recs_menu()
        self.menu.append(self.avail_recs_menu)

        self.menu.append(ui.Text_Input('destination_dir', self, label='Destination directory', setter=self.set_dest_dir))
        self.menu.append(ui.Button('Export selected', self.queue_selected))
        self.menu.append(ui.Button('Clear search results', self._clear_avail))
        self.menu.append(ui.Separator())
        self.menu.append(ui.Button('Cancel all exports', self.cancel_all))


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

    def queue_selected(self):
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

                if out_file_path in self.queued_exports or out_file_path in self.active_exports:
                    logger.error("This export setting would try to save {} at least twice please rename dirs to prevent this. Skipping recording.".format(out_file_path))
                    continue

                self.notify_all({'subject': 'start_plugin', 'name': 'Batch_Export',
                                'args': {'out_file_path': out_file_path,
                                         'rec_dir': avail['source'],
                                         'frames_to_export': frames_to_export}})
                self.available_exports.remove(avail)
                self.queued_exports.append(out_file_path)
        self._update_avail_recs_menu()

    def start_export(self, queued):
        self.notify_all({'subject': 'batch_export.should_start', 'out_file_path': queued})
        self.active_exports.append(queued)
        self.queued_exports.remove(queued)

    def on_notify(self, n):
        if n['subject'] in ('batch_export.canceled', 'batch_export.completed'):
            if n['out_file_path'] in self.queued_exports:
                self.queued_exports.remove(n['out_file_path'])
            if n['out_file_path'] in self.active_exports:
                self.active_exports.remove(n['out_file_path'])

        # Add queued exports to active queue
        for queued in self.queued_exports[:self.worker_count - len(self.active_exports)]:
            self.start_export(queued)


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

    def cancel_all(self):
        self.notify_all({'subject': 'batch_export.should_cancel', 'remove_menu': True})

    def cleanup(self):
        self.cancel_all()
        if self.search_task:
            self.search_task.cancel()

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
