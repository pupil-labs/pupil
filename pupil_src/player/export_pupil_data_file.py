'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import sys, os, platform
import numpy as np
import csv
from collections import MutableMapping

from pyglui import ui

from plugin import Plugin

#logging
import logging
logger = logging.getLogger(__name__)


class Export_Pupil_Data_File(Plugin):
    """docstring for Export_Pupil_data
    Export `pupil_data` into other formats (e.g. `.csv` or `.npy`)
    """
    def __init__(self, g_pool):
        super(Export_Pupil_Data_File, self).__init__(g_pool)
        
        # initialize empty menu
        self.menu = None
        self.export_csv = True
        self.export_npy = False

    def update(self,frame,events):
        pass

    def init_gui(self):
        # initialize the menu
        self.menu = ui.Scrolling_Menu('Export Pupil Data Files')
        # add menu to the window
        self.g_pool.gui.append(self.menu)
        self.menu.append(ui.Button('Close',self.unset_alive))

        self.menu.append(ui.Info_Text('Select the file format(s) and press the export button.'))
        self.menu.append(ui.Switch('export_csv',self,label="export .csv file"))
        self.menu.append(ui.Switch('export_npy',self,label="export .npy file"))
        self.menu.append(ui.Button('export pupil data',self.export_data_file))

    def export_data_file(self):
        # todo - data exports should be a separate process - like exporter launcher
        # todo - add progress slider like in export launcher
        
        self.analytics_dir = os.path.join(self.g_pool.rec_dir,"analytics")
        try:
            os.mkdir(os.path.expanduser(self.analytics_dir))
        except OSError, e:
            # if folder already exists, it will not be overwritten an OSError will be thrown
            logger.debug("Adding export to existing 'analytics' directory.")

        if self.export_csv:
            self.write_dicts_to_csv(self.g_pool.pupil_list,"pupil_positions.csv")
            self.write_dicts_to_csv(self.g_pool.gaze_list,"gaze_positions.csv")
            self.write_dicts_to_csv(self.add_frame_number(self.g_pool.pupil_positions_by_frame),"pupil_positions_by_frame.csv")
            self.write_dicts_to_csv(self.add_frame_number(self.g_pool.gaze_positions_by_frame),"gaze_positions_by_frame.csv")

        if self.export_npy:
            # not yet implemented
            pass

    def write_dicts_to_csv(self, data, filename):
        data_flat = [self.flatten_dict(i) for i in data]
        with open(os.path.join(self.analytics_dir, filename),"wb") as csv_file:
            csv_writer = csv.DictWriter(csv_file,delimiter=",",fieldnames=data_flat[0])
            csv_writer.writeheader()
            for i in data_flat:
                csv_writer.writerow(i)
        logger.debug("Success - finished writing %s" %(filename))        

    def flatten_dict(self, d, parent_key='', sep='_'):
        '''
        source: http://codereview.stackexchange.com/a/21045
        returns a flattened (non-nested) dictionary given nested dict
        prepends names of parent keys to nested child keys 
        '''
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def add_frame_number(self, correlated_data):
        frame_items = []
        for idx,frame_data in enumerate(correlated_data):
            for dict_item in frame_data:
                dict_item.update({"frame": idx})
                frame_items.append(dict_item)

        return frame_items


    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    def unset_alive(self):
        self.alive = False

    def gl_display(self):
        pass

    def get_init_dict(self):
        return {}

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.deinit_gui()

