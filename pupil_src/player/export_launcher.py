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

import logging
logger = logging.getLogger(__name__)

from ctypes import c_bool, c_int,create_string_buffer
from multiprocessing import Process
from multiprocessing.sharedctypes import RawValue

from exporter import export

class Export_Launcher(Plugin):
    """docstring for Export_Launcher
    this plugin can export the video in a seperate process using exporter

    """
    def __init__(self, g_pool,data_dir):
        super(Export_Launcher, self).__init__()
        self.g_pool = g_pool
        self.data_dir = data_dir

        self.new_export = None
        self.active_exports = []


        atb_label = "exporter"
        atb_pos = 10,320


        self._bar = atb.Bar(name =self.__class__.__name__, label=atb_label,
            help="export vizualization video", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.1, size=(300, 100))


        self.update_bar()



    def update_bar(self):
        if self._bar:
            self._bar.clear()

        self._bar.rec_name = create_string_buffer("world_viz.avi",512)
        self._bar.add_var('export name',self._bar.rec_name)
        self._bar.add_button('new export',self.add_export)

        for job,i in zip(self.active_exports,range(len(self.active_exports))):

            self._bar.add_var("%s_progess"%i,create_string_buffer(512),
                            getter= self.atb_progess,
                            data = self.active_exports[i],
                            label='progess',
                            group=str(i),
                            )
            self._bar.add_var("%s_terminate"%i,job.should_terminate,group=str(i),label='cancel')

    def atb_progess(self,job):
        return create_string_buffer("%s / %s" %(job.current_frame.value,job.frames_to_export.value),512)

    def add_export(self):
        logger.debug("Adding new export.")
        should_terminate = RawValue(c_bool,False)
        frames_to_export  = RawValue(c_int,0)
        current_frame = RawValue(c_int,0)

        data_dir = self.data_dir
        start_frame=None
        end_frame=None
        plugins=[]

        out_file_path=self._bar.rec_name.value

        process = Process(target=export, args=(should_terminate,frames_to_export,current_frame, data_dir,start_frame,end_frame,plugins,out_file_path))
        process.should_terminate = should_terminate
        process.frames_to_export = frames_to_export
        process.current_frame = current_frame
        self.new_export = process

    def launch_export(self, new_export):
        logger.debug("Stating new process %s" %new_export)
        new_export.start()
        self.active_exports.append(new_export)
        self.update_bar()

    def update(self,frame,recent_pupil_positions,events):
        if self.new_export:
            self.launch_export(self.new_export)
            self.new_export = None

    def gl_display(self):
        pass




'''
exporter

    - is like a small player
    - launched with args:
        data folder
        start,end frame (trim marks)
        plugins loaded and their config
            - how to do this? 1) load the plugin instance as a whole?
                              2) create a plugin contructor based on a string or something similar?

    - can be used by batch or by player
    - communicates with progress (shared int) and terminate (shared bool)
    - can abort on demand leaving nothing behind
'''

