'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
import csv
import numpy as np
import cv2
import logging
from itertools import chain
from math import atan, tan
from methods import denormalize
from plugin import Plugin
from pyglui import ui
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

        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Info_Text('This plugin detects fixations based on a dispersion threshold in terms of degrees of visual angle. It also uses a min duration threshold.'))
        self.menu.append(ui.Slider('min_duration',self,min=0.0,step=0.05,max=1.0,label='duration threshold'))
        self.menu.append(ui.Slider('max_dispersion',self,min=0.0,step=0.05,max=3.0,label='dispersion threshold'))
        self.menu.append(ui.Button('Run fixation detector',self._classify))
        self.menu.append(ui.Button('Export fixations',self.export_fixations))
        self.menu.append(ui.Switch('show_fixations',self,label='Show fixations'))
        self.menu.append(ui.Slider('h_fov',self,min=5,step=1,max=180,label='horizontal FOV of scene camera',setter=set_h_fov))
        self.menu.append(ui.Slider('v_fov',self,min=5,step=1,max=180,label='vertical FOV of scene camera',setter=set_v_fov))

    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None

    ###todo setters with delay trigger

    def on_notify(self,notification):
        if notification['subject'] == 'gaze_positions_changed':
            logger.info('Gaze postions changed. Recalculating.')
            self._classify()
        elif notification['subject'] == 'fixations_should_recalculate':
            self._classify()


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

        gaze_data = list(chain(*self.g_pool.gaze_positions_by_frame))


        sample_threshold = self.min_duration * 3 *.3 #lets assume we need data for at least 30% of the duration
        dispersion_threshold = self.max_dispersion
        duration_threshold = self.min_duration
        self.notify_all({'subject':'fixations_changed'})

        def dist_deg(p1,p2):
            return np.sqrt(((p1[0]-p2[0])*self.h_fov)**2+((p1[1]-p2[1])*self.v_fov)**2)

        fixations = []
        try:
            fixation_support = [gaze_data.pop(0)]
        except IndexError:
            logger.warning("This recording has no gaze data. Aborting")
            return
        while True:
            fixation_centroid = sum([p['norm_pos'][0] for p in fixation_support])/len(fixation_support),sum([p['norm_pos'][1] for p in fixation_support])/len(fixation_support)
            dispersion = max([dist_deg(fixation_centroid,p['norm_pos']) for p in fixation_support])

            if dispersion < dispersion_threshold and gaze_data:
                #so far all samples inside the threshold, lets add a new candidate
                fixation_support += [gaze_data.pop(0)]
            else:
                if gaze_data:
                    #last added point will break dispersion threshold for current candidate fixation. So we conclude sampling for this fixation.
                    last_sample = fixation_support.pop(-1)
                if fixation_support:
                    duration = fixation_support[-1]['timestamp'] - fixation_support[0]['timestamp']
                    if duration > duration_threshold and len(fixation_support) > sample_threshold:
                        #long enough for fixation: we classifiy this fixation candidate as fixation
                        #calculate character of fixation
                        fixation_centroid = sum([p['norm_pos'][0] for p in fixation_support])/len(fixation_support),sum([p['norm_pos'][1] for p in fixation_support])/len(fixation_support)
                        dispersion = max([dist_deg(fixation_centroid,p['norm_pos']) for p in fixation_support])
                        confidence = sum(g['confidence'] for g in fixation_support)/len(fixation_support)

                        # avg pupil size  = mean of (mean of pupil size per gaze ) for all gaze points of support
                        avg_pupil_size =  sum([sum([p['diameter'] for p in g['base']])/len(g['base']) for g in fixation_support])/len(fixation_support)
                        new_fixation = {'id': len(fixations),
                                        'norm_pos':fixation_centroid,
                                        'gaze':fixation_support,
                                        'duration':duration,
                                        'dispersion':dispersion,
                                        'start_frame_index':fixation_support[0]['index'],
                                        'end_frame_index':fixation_support[-1]['index'],
                                        'pix_dispersion':dispersion*self.pix_per_degree,
                                        'timestamp':fixation_support[0]['timestamp'],
                                        'pupil_diameter':avg_pupil_size,
                                        'confidence':confidence}
                        fixations.append(new_fixation)
                if gaze_data:
                    #start a new fixation candite
                    fixation_support = [last_sample]
                else:
                    break

        self.fixations = fixations
        #gather some statisics for debugging and feedback.
        total_fixation_time  = sum([f['duration'] for f in fixations])
        total_video_time = self.g_pool.timestamps[-1]- self.g_pool.timestamps[0]
        fixation_count = len(fixations)
        logger.info("detected %s Fixations. Total duration of fixations: %0.2fsec total time of video %0.2fsec "%(fixation_count,total_fixation_time,total_video_time))


        # now lets bin fixations into frames. Fixations may be repeated this way as they span muliple frames
        fixations_by_frame = [[] for x in self.g_pool.timestamps]
        for f in fixations:
            for idx in range(f['start_frame_index'],f['end_frame_index']+1):
                fixations_by_frame[idx].append(f)

        self.g_pool.fixations_by_frame = fixations_by_frame


    def export_fixations(self):
        #todo

        in_mark = self.g_pool.trim_marks.in_mark
        out_mark = self.g_pool.trim_marks.out_mark


        """
        between in and out mark

            fixation report:
                - fixation detection method and parameters
                - fixation count

            fixation list:
                id | start_timestamp | duration | start_frame_index | end_frame_index | dispersion | avg_pupil_size | confidence

        """

        if not self.fixations:
            logger.warning('No fixations in this recording nothing to export')
            return

        fixations_in_section = chain(*self.g_pool.fixations_by_frame[slice(in_mark,out_mark)])
        fixations_in_section = dict([(f['id'],f) for f in fixations_in_section]).values() #remove dublicates
        fixations_in_section.sort(key=lambda f:f['id'])
        metrics_dir = os.path.join(self.g_pool.rec_dir,"metrics_%s-%s"%(in_mark,out_mark))
        logger.info("exporting metrics to %s"%metrics_dir)
        if os.path.isdir(metrics_dir):
            logger.info("Will overwrite previous export for this section.")
        else:
            try:
                os.mkdir(metrics_dir)
            except:
                logger.warning("Could not make metrics dir %s!"%metrics_dir)
                return


        with open(os.path.join(metrics_dir,'fixations.csv'),'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(('id','start_timestamp','duration','start_frame','end_frame','norm_pos_x','norm_pos_y','dispersion','avg_pupil_size','confidence'))
            for f in fixations_in_section:
                csv_writer.writerow( ( f['id'],f['timestamp'],f['duration'],f['start_frame_index'],f['end_frame_index'],f['norm_pos'][0],f['norm_pos'][1],f['dispersion'],f['pupil_diameter'],f['confidence'] ) )
            logger.info("Created 'fixations.csv' file.")

        with open(os.path.join(metrics_dir,'fixation_report.csv'),'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(('fixation classifier','Dispersion_Duration'))
            csv_writer.writerow(('max_dispersion','%0.3f deg'%self.max_dispersion) )
            csv_writer.writerow(('min_duration','%0.3f sec'%self.min_duration) )
            csv_writer.writerow((''))
            csv_writer.writerow(('fixation_count',len(fixations_in_section)))
            logger.info("Created 'fixation_report.csv' file.")



    def update(self,frame,events):
        events['fixations'] = self.g_pool.fixations_by_frame[frame.index]
        if self.show_fixations:
            for f in self.g_pool.fixations_by_frame[frame.index]:
                x = int(f['norm_pos'][0]*self.img_size[0])
                y = int((1-f['norm_pos'][1])*self.img_size[1])
                transparent_circle(frame.img, (x,y), radius=f['pix_dispersion'], color=(.5, .2, .6, .7), thickness=-1)
                cv2.putText(frame.img,'%i'%f['id'],(x+20,y), cv2.FONT_HERSHEY_DUPLEX,0.8,(255,150,100))
                # cv2.putText(frame.img,'%i - %i'%(f['start_frame_index'],f['end_frame_index']),(x,y), cv2.FONT_HERSHEY_DUPLEX,0.8,(255,150,100))

    def close(self):
        self.alive = False


    def get_init_dict(self):
        return {'max_dispersion': self.max_dispersion, 'min_duration':self.min_duration, 'h_fov':self.h_fov, 'v_fov': self.v_fov,'show_fixations':self.show_fixations}

    def cleanup(self):
        self.deinit_gui()

