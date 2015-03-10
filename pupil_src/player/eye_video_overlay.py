'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys, os,platform
import cv2
import numpy as np
from file_methods import Persistent_Dict
from pyglui import ui
from methods import normalize,denormalize
from glfw import *
from plugin import Plugin
from glob import glob

#logging
import logging
logger = logging.getLogger(__name__)

class Eye_Video_Overlay(Plugin):
    """docstring
    """
    def __init__(self,g_pool,menu_conf={}):
        super(Eye_Video_Overlay, self).__init__(g_pool)
        self.order = .2
        self.data_dir = g_pool.rec_dir
        self.menu_conf = menu_conf

        meta_info_path = self.data_dir + "/info.csv"

        #parse info.csv file
        with open(meta_info_path) as info:
            meta_info = dict( ((line.strip().split('\t')) for line in info.readlines() ) )
        rec_version = meta_info["Capture Software Version"]
        rec_version_float = int(filter(type(rec_version).isdigit, rec_version)[:3])/100. #(get major,minor,fix of version)
        eye_mode = meta_info["Eye Mode"]

        if rec_version_float < 0.4:
            required_files = ['eye.avi','eye_timestamps.npy']
            eye0_video_path = os.path.join(rec_dir,required_files[0])
            eye0_timestamps_path = os.path.join(rec_dir,required_files[1]) 
        else:
            required_files = ['eye0.mkv','eye0_timestamps.npy']
            eye0_video_path = os.path.join(rec_dir,required_files[0])
            eye0_timestamps_path = os.path.join(rec_dir,required_files[1])
            if eye_mode == 'binocular':
                required_files += ['eye1.mkv','eye1_timestamps.npy']
                eye1_video_path = os.path.join(rec_dir,required_files[2])
                eye1_timestamps_path = os.path.join(rec_dir,required_files[3])
        
        # check to see if eye videos exist
        for f in required_files:
            if not os.path.isfile(os.path.join(rec_dir,f)):
                logger.debug("Did not find required file: ") %(f, rec_dir)
                self.cleanup() # early exit -- no required files

        logger.debug("%s contains eye videos - %s."%(rec_dir,required_files))

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Eye Video Overlay')
        # load the configuration of last session
        self.menu.configuration = self.menu_conf
        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self._update_gui()

    def unset_alive(self):
        self.alive = False

    def _update_gui(self):
        self.menu.elements[:] = []

        self.menu.append(ui.Info_Text('Supply export video recording name. The export will be in the recording dir. If you give a path the export will end up there instead.'))
        self.menu.append(ui.Text_Input('rec_name',self,label='export name'))
        self.menu.append(ui.Info_Text('Select your export frame range using the trim marks in the seek bar.'))
        self.menu.append(ui.Text_Input('in_mark',getter=self.g_pool.trim_marks.get_string,setter=self.g_pool.trim_marks.set_string,label='frame range to export'))
        self.menu.append(ui.Button('new export',self.add_export))

        self.menu.append(ui.Button('close',self.unset_alive))

   def deinit_gui(self):
        if self.menu:
            self.menu_conf = self.menu.configuration
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        if self.menu:
            return {'menu_conf':self.menu.configuration}
        else:
            return {'menu_conf':self.menu_conf}

    def update(self,frame,events):
        # synchronize timestamps with world timestamps
        pass


    def gl_display(self):
        # update the eye texture 
        pass

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()

        