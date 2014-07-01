'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


import logging
logger = logging.getLogger(__name__)

from plugin import Plugin
from offline_marker_detector import Offline_Marker_Detector
from ctypes import c_int

class Marker_Auto_Trim_Marks(Plugin):
    """docstring for Marker_Auto_Trim_Marks:

    Using markers and this plugin sections within one recording can be sliced and autoexported:

    Show marker #x for more than 3 seconds to mark the beginning of a export section
    Show marker #y for more than 3 seconds to mark the end of an export section

    Marker presence is determined by the visibility of a marker for more than 50percent withing 6 seconds.

    This plugin depends on the offline marker tracker plugin to be loaded.

    """
    def __init__(self, g_pool,gui_settings={'pos':(220,400),'size':(300,100),'iconified':False}):
        super(Marker_Auto_Trim_Marks, self).__init__()
        self.g_pool = g_pool
        self.gui_settings = gui_settings

        self.in_marker_id = c_int(0)
        self.out_marker_id = c_int(1)
        self.active_section = c_int(0)
        self.sections = None


    def init_gui(self):
        import atb
        pos = self.gui_settings['pos']
        atb_label = "Marker Auto Trim Marks"
        self._bar = atb.Bar(name =self.__class__.__name__+str(id(self)), label=atb_label,
            help="circle", color=(50, 150, 50), alpha=50,
            text='light', position=pos,refresh=.1, size=self.gui_settings['size'])

        self._bar.iconified = self.gui_settings['iconified']
        self._bar.add_button('close',self.unset_alive)



    def unset_alive(self):
        self.alive = False

    def video_export(self):
        pass

    def surface_export(self):
        pass

    def get_init_dict(self):
        d = {}
        if hasattr(self,'_bar'):
            gui_settings = {'pos':self._bar.position,'size':self._bar.size,'iconified':self._bar.iconified}
            d['gui_settings'] = gui_settings

        return d

    def update_bar_indicator(self,status):
        if status:
            self._bar.color = (50, 50, 50)
            self._bar.label = "Marker Auto Trim Marks"
        else:
            self._bar.color = (250, 50, 50)
            self._bar.label = "Marker Auto Trim Marks: Turn on Offline_Marker_Detector!"


    def update(self,frame,recent_pupil_positions,events):
        if self.sections == None:
            plugins = [p for p in self.g_pool.plugins if isinstance(p,Offline_Marker_Detector)]
            if plugins:
                marker_tracker_plugin = plugins[0]

            else:
                self.update_bar_indicator(False)
                self._bar.color = (250, 50, 50)
                self._bar.label = "Marker Auto Trim Marks: Turn on Offline_Marker_Detector!"


                return

            if marker_tracker_plugin.cache.complete  and 0:
                self._bar.label = "Marker Auto Trim Marks"
                self._bar.color = (50, 50, 50)
                self._bar.add_var('IN marker id',self.in_marker_id,max=63,min=0)
                self._bar.add_var('OUT marker id',self.out_marker_id,max=63,min=0)
                self._bar.add_var("active section", self.active_section)
                self._bar.add_button("  activate section  ",self.activate_section)
                self._bar.add_button("  video export all sections   ",self.video_export)
                self._bar.add_button("   surface export all sections   ",self.surface_export)


                # now lets go through the cache and make in and out events:
                in_idx = []
                out_idx = []
                in_id = self.in_marker_id.value
                out_id = self.out_marker_id.value

                for idx,frame in enumerate(marker_tracker_plugin.cache):
                    # marker = {'id':msg,'verts':r,'verts_norm':r_norm,'centroid':centroid,"frames_since_true_detection":0}
                    for marker in frame:
                        if marker['id'] == in_id:
                            in_idx.append(idx)
                        if marker['id'] == out_id:
                            out_idx.append(idx)

                self.sections = []
                inside = False


            else:
                self._bar.label = "Marker Auto Trim Marks: Waiting for Cacher to finish"
                self._bar.color = (100, 100, 50)







    def cleanup(self):
        self._bar.destroy()
