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
    from billiard import Process,forking_enable
    from billiard.sharedctypes import RawValue
else:
    from multiprocessing import Process
    forking_enable = lambda x: x #dummy fn
    from multiprocessing.sharedctypes import RawValue

from exporter import export

def verify_out_file_path(out_file_path,data_dir):
    #Out file path verification
    if not out_file_path:
        out_file_path = os.path.join(data_dir, "world_viz.avi")
    else:
        file_name =  os.path.basename(out_file_path)
        dir_name = os.path.dirname(out_file_path)
        if not dir_name:
            dir_name = data_dir
        if not file_name:
            file_name = 'world_viz.avi'
        out_file_path = os.path.expanduser(os.path.join(dir_name,file_name))

    out_file_path = avoid_overwrite(out_file_path)
    if os.path.isfile(out_file_path):
        logger.warning("Video out file already exsists. I will overwrite!")
        os.remove(out_file_path)
    logger.debug("Saving Video to %s"%out_file_path)

    return out_file_path


def avoid_overwrite(out_file_path):
    if os.path.isfile(out_file_path):
        # let append something unique
        out_file_path,ext = os.path.splitext(out_file_path)
        out_file_path += str(int(time.time())) + '.avi'
    return out_file_path

class Export_Launcher(Plugin):
    """docstring for Export_Launcher
    this plugin can export the video in a seperate process using exporter

    """
    def __init__(self, g_pool,data_dir,frame_count):
        super(Export_Launcher, self).__init__()
        self.g_pool = g_pool
        self.data_dir = data_dir

        self.new_export = None
        self.exports = []
        # default_path = verify_out_file_path("world_viz.avi",data_dir)
        default_path = "world_viz.avi"

        self.rec_name = create_string_buffer(default_path,512)


    def init_gui(self):

        atb_label = "Export Recording"
        atb_pos = 320,10

        self._bar = atb.Bar(name =self.__class__.__name__, label=atb_label,
            help="export vizualization video", color=(50, 100, 100), alpha=100,
            text='light', position=atb_pos,refresh=.1, size=(300, 150))


        self.update_bar()


    def update_bar(self):
        if self._bar:
            self._bar.clear()

        self._bar.add_var('export name',self.rec_name, help="Supply export video recording name. The export will be in the recording dir. If you give a path the export will end up there instead.")
        self._bar.add_var('start frame',vtype=c_int,getter=self.g_pool.trim_marks.atb_get_in_mark,setter= self.g_pool.trim_marks.atb_set_in_mark, help="Supply start frame no. Negative numbers will count from the end. The behaves like python list indexing")
        self._bar.add_var('end frame',vtype=c_int,getter=self.g_pool.trim_marks.atb_get_out_mark,setter= self.g_pool.trim_marks.atb_set_out_mark,help="Supply end frame no. Negative numbers will count from the end. The behaves like python list indexing")
        self._bar.add_button('new export',self.add_export)

        for job,i in zip(self.exports,range(len(self.exports)))[::-1]:

            self._bar.add_var("%s_out_file"%i,create_string_buffer(512),
                            getter= self.atb_out_file_path,
                            data = self.exports[i],
                            label='file location:',
                            group=str(i),
                            )
            self._bar.add_var("%s_progess"%i,create_string_buffer(512),
                            getter= self.atb_progress,
                            data = self.exports[i],
                            label='progess',
                            group=str(i),
                            )
            self._bar.add_var("%s_terminate"%i,job.should_terminate,group=str(i),label='cancel',help="Cancel export.")

    def atb_progress(self,job):
        if job.current_frame.value == job.frames_to_export.value:
            return create_string_buffer("Done",512)
        return create_string_buffer("%s / %s" %(job.current_frame.value,job.frames_to_export.value),512)

    def atb_out_file_path(self,job):
        return create_string_buffer(job.out_file_path,512)

    def add_export(self):
        # on MacOS we will not use os.fork, elsewhere this does nothing.
        forking_enable(0)

        logger.debug("Adding new export.")
        should_terminate = RawValue(c_bool,False)
        frames_to_export  = RawValue(c_int,0)
        current_frame = RawValue(c_int,0)

        data_dir = self.data_dir
        start_frame= self.g_pool.trim_marks.in_mark
        end_frame= self.g_pool.trim_marks.out_mark+1 #end_frame is exclusive
        plugins = []

        # Here we make clones of every plugin that supports it.
        # So it runs in the current config when we lauch the exporter.
        for p in self.g_pool.plugins:
            try:
                p_initializer = p.get_class_name(),p.get_init_dict()
                plugins.append(p_initializer)
            except AttributeError:
                pass

        out_file_path=verify_out_file_path(self.rec_name.value,self.data_dir)
        process = Process(target=export, args=(should_terminate,frames_to_export,current_frame, data_dir,start_frame,end_frame,plugins,out_file_path))
        process.should_terminate = should_terminate
        process.frames_to_export = frames_to_export
        process.current_frame = current_frame
        process.out_file_path = out_file_path
        self.new_export = process

    def launch_export(self, new_export):

        logger.debug("Starting export as new process %s" %new_export)
        new_export.start()
        self.exports.append(new_export)
        self.update_bar()

    def update(self,frame,recent_pupil_positions,events):
        if self.new_export:
            self.launch_export(self.new_export)
            self.new_export = None


    def gl_display(self):
        pass


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happends either voluntary or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self._bar.destroy()
