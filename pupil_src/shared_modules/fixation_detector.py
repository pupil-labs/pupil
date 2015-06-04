'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import numpy as np
from math import sqrt
import cv2
import logging
from itertools import chain
from math import atan, tan
from methods import denormalize
from plugin import Plugin
from pyglui import ui
from gl_utils.utils import make_coord_system_pixel_based, draw_gl_polyline,\
    draw_gl_polyline_norm
from player_methods import transparent_circle
# logging
logger = logging.getLogger(__name__)

class Fixation_Detector(Plugin):
    """ base class for different fixation detection algorithms """
    def __init__(self, g_pool):
        super(Fixation_Detector, self).__init__(g_pool)


class Dispersion_Duration_Fixation_Detector(Fixation_Detector):
    '''
    This plugin classifies fixations and saccades by measuring dispersion and duration of gaze points


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
    def __init__(self,g_pool,max_dispersion = 1.0,min_duration = 0.15,h_fov=78, v_fov=50,show_fixations = False):
        super(Dispersion_Duration_Fixation_Detector, self).__init__(g_pool)
        self.min_duration = min_duration
        self.max_dispersion = max_dispersion
        self.h_fov = h_fov
        self.v_fov = v_fov
        self.show_fixations = show_fixations

        self.pix_per_degree = float(self.g_pool.capture.frame_size[0])/h_fov
        self.img_size = self.g_pool.capture.frame_size
        self.fixations_to_display = []
        self._classify()

    def init_gui(self):
        self.menu = ui.Scrolling_Menu('Fixation Detector')
        self.g_pool.gui.append(self.menu)

        def set_h_fov(new_fov):
            self.h_fov = new_fov
            self.pix_per_degree = float(self.g_pool.capture.frame_size[0])/new_fov

        def set_v_fov(new_fov):
            self.v_fov = new_fov
            self.pix_per_degree = float(self.g_pool.capture.frame_size[1])/new_fov

        self.menu.append(ui.Info_Text('This plugin detects fixations based on a dispersion threshold in terms of degrees of visual angle. It also uses a min duration threshold.'))
        self.menu.append(ui.Slider('min_duration',self,min=0.0,step=0.05,max=1.0,label='duration threshold'))
        self.menu.append(ui.Slider('max_dispersion',self,min=0.0,step=0.05,max=3.0,label='dispersion threshold'))
        self.menu.append(ui.Button('Run fixation detector',self._classify))
        self.menu.append(ui.Switch('show_fixations',self,label='Show fixations'))
        self.menu.append(ui.Slider('h_fov',self,min=5,step=1,max=180,label='horizontal FOV of scene camera',setter=set_h_fov))
        self.menu.append(ui.Slider('v_fov',self,min=5,step=1,max=180,label='vertical FOV of scene camera',setter=set_v_fov))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    # def _velocity(self):
    #     """
    #     distance petween gaze points dn = gn-1 - gn
    #     dt = tn-1 - tn
    #     velocity vn = gn/tn
    #     """
    #     gaze_data = chain(*self.g_pool.gaze_positions_by_frame)
    #     gaze_positions = np.array([gp['norm_pos'] for gp in gaze_data])
    #     for gp0,gp1 in zip(gaze_data[:-1],gaze_data[1:]):
    #         angular_dist = self.dist_deg(gp0['norm_pos'],gp1['norm_pos'])
    #         angular_vel = angular_dist/(gp1['timestamp']-gp0['timestamp']

    def _classify(self):
        '''
        classify fixations
        '''
        gaze_data = chain(*self.g_pool.gaze_positions_by_frame)
        #filter out  below threshold confidence mesurements
        gaze_data = filter(lambda g: g['confidence'] > self.g_pool.pupil_confidence_threshold, gaze_data)



        sample_threshold = self.min_duration * 3 *.3 #lets assume we need data for at least 30% of the duration
        dispersion_threshold = self.max_dispersion
        duration_threshold = self.min_duration


        def dist_deg(p1,p2):
            return sqrt(((p1[0]-p2[0])*self.h_fov)**2+((p1[1]-p2[1])*self.v_fov)**2)

        fixations = []
        fixation_support = [gaze_data.pop(0)]


        while gaze_data:
            fixation_centroid = sum([p['norm_pos'][0] for p in fixation_support])/len(fixation_support),sum([p['norm_pos'][1] for p in fixation_support])/len(fixation_support)
            dispersion = max([dist_deg(fixation_centroid,p['norm_pos']) for p in fixation_support])

            if dispersion < dispersion_threshold:
                #so far all samples inside the threshold, lets add a new canditate
                fixation_support.append(gaze_data.pop(0))
            else:
                #last added point will break dispersion threshold for current candite fixation. So we conclude sampling for this fixation
                last_sample = fixation_support.pop(-1)
                duration = fixation_support[-1]['timestamp'] - fixation_support[0]['timestamp']
                if duration > duration_threshold and len(fixation_support) > sample_threshold:
                    #long enough for fixation: we classifiy this fixation canditae as fixation
                    fixation_centroid = sum([p['norm_pos'][0] for p in fixation_support])/len(fixation_support),sum([p['norm_pos'][1] for p in fixation_support])/len(fixation_support)
                    dispersion = max([dist_deg(fixation_centroid,p['norm_pos']) for p in fixation_support])
                    new_fixation = {'id': len(fixations),'norm_pos':fixation_centroid,'gaze':fixation_support, 'duration':duration,'dispersion':dispersion, 'pix_dispersion':dispersion*self.pix_per_degree, 'start_timestamp':fixation_support[0]['timestamp']}
                    fixations.append(new_fixation)
                #start a new fixation candite
                fixation_support = [last_sample]


        logger.debug("detected %s Fixations"%len(fixations))
        self.fixations = fixations[:] #keep a copy because we destroy our list below.

        # now lets bin fixations into frames. Fixations may be repeated this way as they span muliple frames
        fixations_by_frame = [[] for x in self.g_pool.timestamps]
        index = 0
        f = fixations.pop(0)
        while fixations:
            try:
                t = self.g_pool.timestamps[index]
            except IndexError:
                #reached end of ts list
                break
            if f['start_timestamp'] > t:
                #fixation in the future, lets move forward in time.
                index += 1
            elif  f['start_timestamp']+f['duration'] > t:
                # fixation during this frame
                fixations_by_frame[index].append(f)
                index += 1
            else:
                #fixation in the past, get new one and check again
                f = fixations.pop(0)

        self.fixations_by_frame = fixations_by_frame


    def update(self,frame,events):
        events['fixations'] = self.fixations_by_frame[frame.index]
        if self.show_fixations:
            self.fixations_to_display = self.fixations_by_frame[frame.index]
            for f in self.fixations_to_display:
                x = int(f['norm_pos'][0]*self.img_size[0])
                y = int((1-f['norm_pos'][1])*self.img_size[1])
                transparent_circle(frame.img, (x,y), radius=f['pix_dispersion'], color=(.5, .2, .6, .7), thickness=-1)
                cv2.putText(frame.img,'%i'%f['id'],(x,y), cv2.FONT_HERSHEY_DUPLEX,0.8,(255,100,100))


    def gl_display(self):
        pass
        # for f in self.fixations_to_display:
            # print f['id'],f['norm_pos']

    def get_init_dict(self):
        return {'max_dispersion': self.max_dispersion, 'min_duration':self.min_duration, 'h_fov':self.h_fov, 'v_fov': self.v_fov,'show_fixations':self.show_fixations}

    def cleanup(self):
        self.deinit_gui()

