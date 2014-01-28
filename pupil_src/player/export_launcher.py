'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Plugin
import numpy as np
import atb
import os

import logging
logger = logging.getLogger(__name__)

from ctypes import c_bool, c_int,create_string_buffer
from multiprocessing import Process
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

    if os.path.isfile(out_file_path):
        logger.warning("Video out file already exsists. I will overwrite!")
        os.remove(out_file_path)
    logger.debug("Saving Video to %s"%out_file_path)

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

        self.rec_name = create_string_buffer("world_viz.avi",512)
        self.start_frame = c_int(0)
        self.end_frame = c_int(frame_count)

        atb_label = "Export Recording"
        atb_pos = 10,220


        self._bar = atb.Bar(name =self.__class__.__name__, label=atb_label,
            help="export vizualization video", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.1, size=(300, 300))


        self.update_bar()



    def update_bar(self):
        if self._bar:
            self._bar.clear()


        self._bar.add_var('export name',self.rec_name)
        self._bar.add_var('start frame',self.start_frame)
        self._bar.add_var('end frame',self.end_frame)
        self._bar.add_button('new export',self.add_export)

        for job,i in zip(self.exports,range(len(self.exports))):

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
            self._bar.add_var("%s_terminate"%i,job.should_terminate,group=str(i),label='cancel')

    def atb_progress(self,job):
        if job.current_frame.value == job.frames_to_export.value:
            return create_string_buffer("Done",512)
        return create_string_buffer("%s / %s" %(job.current_frame.value,job.frames_to_export.value),512)

    def atb_out_file_path(self,job):
        return create_string_buffer(job.out_file_path,512)

    def add_export(self):
        logger.debug("Adding new export.")
        should_terminate = RawValue(c_bool,False)
        frames_to_export  = RawValue(c_int,0)
        current_frame = RawValue(c_int,0)

        data_dir = self.data_dir
        start_frame= self.start_frame.value
        end_frame= self.end_frame.value
        plugins=[]

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

        # for j in self.exports:
        #     if not j.is_alive():
        #         print j.exitcode

    def gl_display(self):
        pass


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happends either voluntary or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self._bar.destroy()
