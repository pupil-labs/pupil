'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import numpy as np
import cv2
import logging
from math import atan, tan
from methods import denormalize
from plugin import Plugin
from pyglui import ui
from gl_utils.utils import make_coord_system_pixel_based, draw_gl_polyline,\
    draw_gl_polyline_norm

# logging
logger = logging.getLogger(__name__)

class Fixation_Detector(Plugin):
    """ base class for different fixation detection algorithms """
    def __init__(self, g_pool):
        super(Fixation_Detector, self).__init__(g_pool)
        
class Dispersion_Fixation_Detector(Fixation_Detector):
    
    """ fixation detection algorithm based on a dispersion threshold """
    def __init__(self, g_pool, dispersion=0.45, h_fov=78, v_fov=50, menu_conf={}):
        super(Fixation_Detector, self).__init__(g_pool)
        self.menu_conf = menu_conf
        self.dispersion = dispersion
        self.h_fov = h_fov
        self.v_fov = v_fov
        self.scene_res = self.g_pool.capture.frame_size
        self.angle_factor = (0.5/tan(np.deg2rad(self.h_fov/2)), 0.5/tan(np.deg2rad(self.v_fov/2)))
        self.fixation = None
        self.gaze_history = []
        
    def init_gui(self):
        self.menu = ui.Growing_Menu('Fixation Detector')
        self.menu.configuration = self.menu_conf
        self.g_pool.sidebar.append(self.menu)
        
        self.menu.append(ui.Info_Text('This plugin detects fixations based on a dispersion threshold in terms of degrees of visual angle'))
        self.menu.append(ui.Slider('dispersion',self,min=0.1,step=0.05,max=1.0,label='dispersion threshold'))
        self.menu.append(ui.Slider('h_fov',self,min=5,step=1,max=180,label='horizontal FOV of scene camera'))
        self.menu.append(ui.Slider('v_fov',self,min=5,step=1,max=180,label='vertical FOV of scene camera'))
        
    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None
    
    def update(self,frame,events):
        for pt in events.get('gaze',[]):
            gaze = pt['norm_pos']
            
            # compute angular distance between previous and current gaze sample, if there is one
            prev_gaze = None
            if len(self.gaze_history) > 0:
                prev_gaze = self.gaze_history[-1]
                angular_dist = self.compute_angular_distance(prev_gaze, gaze)
                
                if angular_dist > self.dispersion:
                    self.fixation = np.mean(np.array(self.gaze_history, dtype = np.float32), axis = 0)
                    # TODO: add fixation to events
                    self.gaze_history = []
            self.gaze_history.append(gaze)
                
    def compute_angular_distance(self, prev_gaze, gaze):
        # move origin to image center
        prev_gaze_x, prev_gaze_y = prev_gaze[0] - 0.5, prev_gaze[1] - 0.5
        gaze_x, gaze_y = gaze[0] - 0.5, gaze[1] - 0.5
        # compute angular coordinates for gaze and previous gaze sample
        angle_prev_x = np.rad2deg(atan(float(prev_gaze_x) / self.angle_factor[0]))
        angle_prev_y = np.rad2deg(atan(float(prev_gaze_y) / self.angle_factor[1]))
        angle_x = np.rad2deg(atan(float(gaze_x) / self.angle_factor[0]))
        angle_y = np.rad2deg(atan(float(gaze_y) / self.angle_factor[1]))
        # return the diagonal angular distance
        return np.linalg.norm([angle_prev_x - angle_x, angle_prev_y - angle_y])
    
    def gl_display(self):
        if self.fixation is not None:
            abs_fixation = denormalize(self.fixation,self.g_pool.capture.frame_size, flip_y=True)
            ellipse = cv2.ellipse2Poly((int(abs_fixation[0]), int(abs_fixation[1])),(25, 25),0,0,360,15)
            draw_gl_polyline(ellipse,(0.,0.,.5,.75),'Polygon')
    
    def get_init_dict(self):
        return {'dispersion': self.dispersion, 'h_fov':self.h_fov, 'v_fov': self.v_fov, 'menu_conf': self.menu.configuration}
    
    def cleanup(self):
        self.deinit_gui()