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
from itertools import chain
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


class Dispersion_Duration_Fixation_Detector(Fixation_Detector):
    '''
    This plugin classifies fixations and saccades by measuring dispersion and duration of gaze points

    Methods of fixation detection are based on prior literature
        (Salvucci & Goldberg, ETRA, 2000) http://www.cs.drexel.edu/~salvucci/publications/Salvucci-ETRA00.pdf
        (Munn et al., APGV, 2008) http://www.cis.rit.edu/vpl/3DPOR/website_files/Munn_Stefano_Pelz_APGV08.pdf
        (Evans et al, JEMR, 2012) http://www.jemr.org/online/5/2/6

    Smooth Pursuit/Ego-motion accounted for by optical flow in Scan Path plugin:
        Reference literature (Kinsman et al. "Ego-motion compensation improves fixation detection in wearable eye tracking," ACM 2011)

    Fixations general knowledge from literature review
        + Goldberg et al. - fixations rarely < 100ms and range between 200ms and 400ms in duration (Irwin, 1992 - fixations dependent on task between 150ms - 600ms)
        + Very short fixations are considered not meaningful for studying behavior - eye+brain require time for info to be registered (see Munn et al. APGV, 2008)
        + Fixations are rarely longer than 800ms in duration
            + Smooth Pursuit is exception and different motif
            + If we do not set a maximum duration, we will also detect smooth pursuit (which is acceptable since we compensate for VOR)
    Terms
        + dispersion (spatial) = how much spatial movement is allowed within one fixation (in visual angular degrees or pixels)
        + duration (temporal) = what is the minimum time required for gaze data to be within dispersion threshold?
        + cohesion (spatial+temporal) = is the cluster of fixations close together

    '''
    def __init__(self,g_pool,min_dispersion = 0.45,min_duration = 0.15,h_fov=78, v_fov=50,show_fixations = False, ):
        self.min_duration = min_duration
        self.min_dispersion = min_dispersion
        self.h_fov = h_fov
        self.v_fov = v_fov
        self.show_fixations = show_fixations

        self.pix_per_degree = float(self.g_pool.capture.frame_size[0])/h_fov

    def init_gui(self):
        self.menu = ui.Growing_Menu('Fixation Detector')
        self.g_pool.sidebar.append(self.menu)

        def set_h_fov(new_fov):
            self.h_fov = new_fov
            self.pix_per_degree = float(self.g_pool.capture.frame_size[0])/new_fov

        def set_v_fov(new_fov):
            self.v_fov = new_fov
            self.pix_per_degree = float(self.g_pool.capture.frame_size[1])/new_fov

        self.menu.append(ui.Info_Text('This plugin detects fixations based on a dispersion threshold in terms of degrees of visual angle. It also uses a min duration threshold.'))
        self.menu.append(ui.Slider('dispersion',self,min=0.1,step=0.05,max=1.0,label='dispersion threshold'))
        self.menu.append(ui.Slider('h_fov',self,min=5,step=1,max=180,label='horizontal FOV of scene camera',setter=set_h_fov))
        self.menu.append(ui.Slider('v_fov',self,min=5,step=1,max=180,label='vertical FOV of scene camera',setter=set_v_fov))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def _classify(self):
        '''
        distance petween gaze points dn = gn-1 - gn
        dt = tn-1 - tn
        velocity vn = gn/tn

        we filter for spikes in velocity
        '''
        gaze_data = chain(*self.g_pool.gaze_positions_by_frame)
        # gaze_positions = np.array([gp['norm_pos'] for gp in gaze_data])

        # for gp0,gp1 in zip(gaze_data[:-1],gaze_data[1:]):
        #     angular_dist = self.dist_deg(gp0['norm_pos'],gp1['norm_pos'])
        #     angular_vel = angular_dist/(gp1['timestamp']-gp0['timestamp']



        fixations = []
        fixation_centroid = 0,0
        fixation_support = []


    def dist_deg(p1,p2):
        return sqrt(((p1[0]-p2[0])*self.h_fov)**2+((p1[1]-p2[1])*self.v_fov)**2)


    def update(self,frame,events):
        pass

    def gl_display(self):
        if self.show_fixations:
            pass

    def get_init_dict(self):
        return {'min_dispersion': self.min_dispersion, 'min_duration':self.min_duration, 'h_fov':self.h_fov, 'v_fov': self.v_fov,'show_fixations':self.show_fixations}

    def cleanup(self):
        self.deinit_gui()


class Dispersion_Fixation_Detector(Fixation_Detector):

    """ fixation detection algorithm based on a dispersion threshold """
    def __init__(self, g_pool, dispersion=0.45, h_fov=78, v_fov=50):
        super(Fixation_Detector, self).__init__(g_pool)
        self.dispersion = dispersion
        self.h_fov = h_fov
        self.v_fov = v_fov
        self.scene_res = self.g_pool.capture.frame_size
        self.angle_factor = (0.5/tan(np.deg2rad(self.h_fov/2)), 0.5/tan(np.deg2rad(self.v_fov/2)))
        self.fixation = None
        self.gaze_history = []

    def init_gui(self):
        self.menu = ui.Growing_Menu('Fixation Detector')
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
        for pt in events.get('gaze_positions',[]):
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
        return {'dispersion': self.dispersion, 'h_fov':self.h_fov, 'v_fov': self.v_fov}

    def cleanup(self):
        self.deinit_gui()