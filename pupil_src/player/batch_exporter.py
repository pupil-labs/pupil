'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Plugin
import numpy as np
import atb
import os,sys, platform
import time

import logging
logger = logging.getLogger(__name__)

from ctypes import c_bool, c_int,create_string_buffer

if platform.system() == 'Darwin':
    from billiard import Process,forking_enable,cpu_count
    from billiard.sharedctypes import RawValue
else:
    from multiprocessing import Process,cpu_count
    forking_enable = lambda x: x #dummy fn
    from multiprocessing.sharedctypes import RawValue

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
        super(Batch_Exporter, self).__init__()
        self.g_pool = g_pool

        self.exports = []
        self.new_exports = []
        self.active_exports = []
        default_path = os.path.expanduser('~/Desktop')
        self.destination_dir = create_string_buffer(default_path,512)
        self.source_dir = create_string_buffer(default_path,512)

        self.run = False
        self.workers = [None for x in range(cpu_count())]
        logger.info("Using a maximum of %s CPUs to process visualizations in parallel..." %cpu_count())

    def init_gui(self):

        atb_label = "Batch Export Recordings"
        atb_pos = 320,310


        self._bar = atb.Bar(name =self.__class__.__name__, label=atb_label,
            help="export vizualization videos", color=(50, 100, 100), alpha=100,
            text='light', position=atb_pos,refresh=.1, size=(300, 150))

        self.update_bar()


    def update_bar(self):
        if self._bar:
            self._bar.clear()

        self._bar.add_var('src_dir',create_string_buffer(512),getter = lambda: self.source_dir, setter=self.set_src_dir,label='recordings src dir')
        self._bar.add_var('dest_dir',create_string_buffer(512),getter = lambda: self.destination_dir, setter=self.set_dest_dir,label='recordings destination dir')
        self._bar.add_button('start',self.start,label='start export')

        for job,i in zip(self.exports,range(len(self.exports)))[::-1]:

            self._bar.add_var("%s_out_file"%i,create_string_buffer(512),
                            getter= self.atb_out_file_path,
                            data = self.exports[i],
                            label='location:',
                            group='Job '+str(i),
                            )
            self._bar.add_var("%s_progess"%i,create_string_buffer(512),
                            getter= self.atb_progress,
                            data = self.exports[i],
                            label='progess',
                            group='Job '+str(i),
                            )
            self._bar.add_var("%s_terminate"%i,job.should_terminate,group='Job '+str(i),label='cancel',help="Cancel export.")

    def atb_progress(self,job):
        return create_string_buffer("%s / %s" %(job.current_frame.value,job.frames_to_export.value),512)

    def atb_out_file_path(self,job):
        return create_string_buffer(job.out_file_path,512)


    def set_src_dir(self,new_dir):
        new_dir = new_dir.value
        self.new_exports = []
        self.exports = []
        new_dir = os.path.expanduser(new_dir)
        if os.path.isdir(new_dir):
            self.source_dir.value = new_dir
            self.new_exports = get_recording_dirs(new_dir)
        else:
            logger.warning('"%s" is not a directory'%new_dir)
            return
        if self.new_exports is []:
            logger.warning('"%s" does not contain recordings'%new_dir)
            return

        self.add_exports()
        self.update_bar()

    def set_dest_dir(self,new_dir):
        new_dir = new_dir.value

        new_dir = os.path.expanduser(new_dir)
        if os.path.isdir(new_dir):
            self.destination_dir.value = new_dir
        else:
            logger.warning('"%s" is not a directory'%new_dir)
            return

        self.exports = []
        self.add_exports()
        self.update_bar()



    def add_exports(self):
        # on MacOS we will not use os.fork, elsewhere this does nothing.
        forking_enable(0)

        outfiles = set()
        for d in self.new_exports:
            logger.debug("Adding new export.")
            should_terminate = RawValue(c_bool,False)
            frames_to_export  = RawValue(c_int,0)
            current_frame = RawValue(c_int,0)
            start_frame = None
            end_frame = None
            data_dir = d
            plugins = []

            # Here we make clones of every plugin that supports it.
            # So it runs in the current config when we lauch the exporter.
            for p in self.g_pool.plugins:
                try:
                    p_initializer = p.get_class_name(),p.get_init_dict()
                    plugins.append(p_initializer)
                except AttributeError:
                    pass

            #make a unique name created from rec_session and dir name
            rec_session, rec_dir = data_dir.rsplit(os.path.sep,2)[1:]
            out_name = rec_session+"_"+rec_dir+".avi"
            out_file_path = os.path.join(self.destination_dir.value,out_name)
            if out_file_path in outfiles:
                logger.error("This export setting would try to save %s at least twice please rename dirs to prevent this. Skipping File"%out_file_path)
            else:
                outfiles.add(out_file_path)
                logger.info("Exporting to: %s"%out_file_path)

                process = Process(target=export, args=(should_terminate,frames_to_export,current_frame, data_dir,start_frame,end_frame,plugins,out_file_path))
                process.should_terminate = should_terminate
                process.frames_to_export = frames_to_export
                process.current_frame = current_frame
                process.out_file_path = out_file_path
                self.exports.append(process)

    def start(self):
        self.active_exports = self.exports[:]
        self.run = True


    def update(self,frame,recent_pupil_positions,events):
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
        self._bar.destroy()
