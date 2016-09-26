'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from . import Base_Manager
from ..source import File_Source, Fake_Source
import os.path

import logging, traceback as tb
logger = logging.getLogger(__name__)

class File_Manager(Base_Manager):

    gui_name = 'Video File Source'
    file_exts = ['.mp4','.mkv','.mov']

    def __init__(self, g_pool, root_folder=None):
        super(File_Manager, self).__init__(g_pool)
        base_dir = self.g_pool.user_dir.rsplit(os.path.sep,1)[0]
        default_rec_dir = os.path.join(base_dir,'recordings')
        self.root_folder = root_folder or default_rec_dir
        self.selected_file = None
        self.eligible_files = [None]

    def init_gui(self):
        from pyglui import ui
        ui_elements = []
        ui_elements.append(ui.Info_Text('Enter a folder to enumerate all eligible video files. Be aware that entering folders with a lot of files can slow down Pupil Capture.'))

        def set_root(folder):
            if not os.path.isdir(folder):
                logger.error('`%s` is not a valid folder path.'%folder)
            else: self.root_folder = folder

        ui_elements.append(ui.Text_Input('root_folder',self,label='Source Folder',setter=set_root))

        def split_enumeration():
            eligible_files = self.enumerate_folder(self.root_folder)
            eligible_files.insert(0, (None, 'Select to activate'))
            return zip(*eligible_files)

        def activate(full_path):
            if not full_path:
                return
            settings = {
                'source_class_name': File_Source.class_name(),
                'frame_size': self.g_pool.capture.frame_size,
                'frame_rate': self.g_pool.capture.frame_rate,
                'source_path': full_path
            }
            self.activate_source(File_Source, settings)

        ui_elements.append(ui.Selector(
            'selected_file',self,
            selection_getter=split_enumeration,
            getter=lambda: None,
            setter=activate,
            label='Video File'
        ))

        self.g_pool.capture_selector_menu.extend(ui_elements)

    def enumerate_folder(self,path):
        eligible_files  = []
        is_eligible = lambda f: os.path.splitext(f)[-1] in self.file_exts
        path = os.path.abspath(os.path.expanduser(path))
        for root,dirs,files in os.walk(path):
            def root_split(file):
                full_p = os.path.join(root,file)
                disp_p = full_p.replace(path,'')
                return (full_p, disp_p)
            eligible_files.extend(map(root_split, filter(is_eligible, files)))
        return eligible_files

    def get_init_dict(self):
        return {'root_folder':self.root_folder}
