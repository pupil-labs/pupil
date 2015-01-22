'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from gl_utils import draw_gl_points_norm
from plugin import Plugin
import numpy as np
from ctypes import c_int,c_float,c_bool
import cv2

from methods import denormalize

class Vis_Cross(Plugin):
    """docstring for DisplayGaze"""
    def __init__(self, g_pool,inner=20,outer=100,color=(1.,.2,.4,.5),thickness=1,gui_settings={'pos':(10,420),'size':(300,100),'iconified':False}):
        super(Vis_Cross, self).__init__(g_pool)
        self.order = .9

        self.gui_settings = gui_settings

        self.inner = c_int(int(inner))
        self.outer = c_int(int(outer))
        self.color = (c_float*4)(*color)
        self.thickness = c_int(int(thickness))


    def update(self,frame,recent_pupil_positions,events):
        color = map(lambda x:int(x*255),self.color)
        color = color[:3][::-1]+color[-1:]
        thickness = self.thickness.value
        inner = self.inner.value
        outer = self.outer.value

        pts = [denormalize(pt['norm_gaze'],frame.img.shape[:-1][::-1],flip_y=True) for pt in recent_pupil_positions if pt['norm_gaze'] is not None]
        for pt in pts:
            lines =  np.array( [((pt[0]-inner,pt[1]),(pt[0]-outer,pt[1])),((pt[0]+inner,pt[1]),(pt[0]+outer,pt[1])) , ((pt[0],pt[1]-inner),(pt[0],pt[1]-outer)) , ((pt[0],pt[1]+inner),(pt[0],pt[1]+outer))],dtype=np.int32 )
            cv2.polylines(frame.img, lines, isClosed=False, color=color, thickness=thickness, lineType=cv2.cv.CV_AA)

    def init_gui(self,pos=None):
        pos = self.gui_settings['pos']
        import atb
        atb_label = "Gaze Cross"
        self._bar = atb.Bar(name =self.__class__.__name__+str(id(self)), label=atb_label,
            help="circle", color=(50, 50, 50), alpha=50,
            text='light', position=pos,refresh=.1, size=self.gui_settings['size'])
        self._bar.iconified = self.gui_settings['iconified']

        self._bar.add_var('color',self.color)
        self._bar.add_var('inner',self.inner, min=0)
        self._bar.add_var('outer',self.outer, min=0)
        self._bar.add_var('thickness',self.thickness,min=1)
        self._bar.add_button('remove',self.unset_alive)

    def unset_alive(self):
        self.alive = False

    def gl_display(self):
        pass

    def get_init_dict(self):
        d = {'inner':self.inner.value,'outer':self.outer.value,'color':self.color[:],'thickness':self.thickness.value}

        if hasattr(self,'_bar'):
            gui_settings = {'pos':self._bar.position,'size':self._bar.size,'iconified':self._bar.iconified}
            d['gui_settings'] = gui_settings

        return d

    def clone(self):
        return Vis_Cross(**self.get_init_dict())


    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have an atb bar or glfw window destroy it here.
        """
        self._bar.destroy()


