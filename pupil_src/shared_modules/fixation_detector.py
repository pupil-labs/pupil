'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''


import os, time
import csv
import numpy as np
import cv2
import logging
from itertools import chain
from math import atan, tan
from operator import itemgetter
from methods import denormalize
from plugin import Plugin, Analysis_Plugin_Base
from pyglui import ui

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def angle_between_normals(v1, v2):
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


class Offline_Base_Fixation_Detector(Analysis_Plugin_Base):
    """ base class for different fixation detection algorithms """
    def __init__(self, g_pool):
        super().__init__(g_pool)


class Gaze_Position_2D_Fixation_Detector(Offline_Base_Fixation_Detector):
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
        super().__init__(g_pool)
        self.min_duration = min_duration
        self.max_dispersion = max_dispersion
        self.h_fov = h_fov
        self.v_fov = v_fov
        self.show_fixations = show_fixations

        self.dispersion_slider_min = 0.
        self.dispersion_slider_max = 3.
        self.dispersion_slider_stp = 0.05

        self.pix_per_degree = float(self.g_pool.capture.frame_size[0])/h_fov
        self.img_size = self.g_pool.capture.frame_size
        self.fixations_to_display = []
        logger.info("Classifying fixations.")
        self.notify_all({'subject':'fixations_should_recalculate','delay':.5})

    @classmethod
    def menu_title(self):
        return 'Gaze Position Dispersion Fixation Detector'

    def init_gui(self):
        self.menu = ui.Scrolling_Menu(self.menu_title())
        self.g_pool.gui.append(self.menu)

        def set_h_fov(new_fov):
            self.h_fov = new_fov
            self.pix_per_degree = float(self.g_pool.capture.frame_size[0])/new_fov
            self.notify_all({'subject':'fixations_should_recalculate','delay':1.})

        def set_v_fov(new_fov):
            self.v_fov = new_fov
            self.pix_per_degree = float(self.g_pool.capture.frame_size[1])/new_fov
            self.notify_all({'subject':'fixations_should_recalculate','delay':1.})

        def set_duration(new_value):
            self.min_duration = new_value
            self.notify_all({'subject':'fixations_should_recalculate','delay':1.})

        def set_dispersion(new_value):
            self.max_dispersion = new_value
            self.notify_all({'subject':'fixations_should_recalculate','delay':1.})


        def jump_next_fixation(_):
            ts = self.last_frame_ts
            for f in self.fixations:
                if f['timestamp'] > ts:
                    self.g_pool.capture.seek_to_frame(f['mid_frame_index'])
                    self.g_pool.new_seek = True
                    return
            logger.error('could not seek to next fixation.')



        self.menu.append(ui.Button('Close',self.close))
        self.menu.append(ui.Info_Text('This plugin detects fixations based on a dispersion threshold in terms of degrees of visual angle. It also uses a min duration threshold.'))
        self.menu.append(ui.Info_Text("Press the export button or type 'e' to start the export."))
        self.menu.append(ui.Slider('min_duration',self,min=0.0,step=0.05,max=1.0,label='Duration threshold',setter=set_duration))
        self.menu.append(ui.Slider('max_dispersion',self,
            min =self.dispersion_slider_min,
            step=self.dispersion_slider_stp,
            max =self.dispersion_slider_max,
            label='Dispersion threshold',
            setter=set_dispersion))
        self.menu.append(ui.Switch('show_fixations',self,label='Show fixations'))
        self.menu.append(ui.Slider('h_fov',self,min=5,step=1,max=180,label='Horizontal FOV of scene camera',setter=set_h_fov))
        self.menu.append(ui.Slider('v_fov',self,min=5,step=1,max=180,label='Vertical FOV of scene camera',setter=set_v_fov))


        self.add_button = ui.Thumb('jump_next_fixation',setter=jump_next_fixation,getter=lambda:False,label=chr(0xf051),hotkey='f',label_font='fontawesome',label_offset_x=0,label_offset_y=2,label_offset_size=-24)
        self.add_button.status_text = 'Next Fixation'
        self.g_pool.quickbar.append(self.add_button)


    def deinit_gui(self):
        if self.menu:
            self.g_pool.gui.remove(self.menu)
            self.menu = None
        if self.add_button:
            self.g_pool.quickbar.remove(self.add_button)
            self.add_button = None
    ###todo setters with delay trigger

    def on_notify(self,notification):
        if notification['subject'] == 'gaze_positions_changed':
            logger.info('Gaze postions changed. Recalculating.')
            self._classify()
        elif notification['subject'] == 'fixations_should_recalculate':
            self._classify()
        elif notification['subject'] == "should_export":
            self.export_fixations(notification['range'],notification['export_dir'])



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

        logger.info("Reclassifying fixations.")
        gaze_data = list(chain(*self.g_pool.gaze_positions_by_frame))

        # lets assume we need data for at least 30% of the duration
        sample_threshold = self.min_duration * self.g_pool.capture.frame_rate * .33
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
                        # long enough for fixation: we classifiy this fixation candidate as fixation
                        # calculate character of fixation
                        fixation_centroid = sum([p['norm_pos'][0] for p in fixation_support])/len(fixation_support),sum([p['norm_pos'][1] for p in fixation_support])/len(fixation_support)
                        dispersion = max([dist_deg(fixation_centroid,p['norm_pos']) for p in fixation_support])
                        confidence = sum(g['confidence'] for g in fixation_support)/len(fixation_support)

                        # avg pupil size  = mean of (mean of pupil size per gaze ) for all gaze points of support
                        avg_pupil_size =  sum([sum([p['diameter'] for p in g['base_data']])/len(g['base_data']) for g in fixation_support])/len(fixation_support)
                        new_fixation = {'topic':'fixation',
                                        'id': len(fixations),
                                        'norm_pos':fixation_centroid,
                                        'base_data':fixation_support,
                                        'duration':duration,
                                        'dispersion':dispersion,
                                        'start_frame_index':fixation_support[0]['index'],
                                        'mid_frame_index':fixation_support[len(fixation_support)//2]['index'],
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
        logger.info("Detected {} Fixations. Total duration of fixations: {:.2f}sec total time of video {:0.2f}sec ".format(fixation_count, total_fixation_time, total_video_time))


        # now lets bin fixations into frames. Fixations may be repeated this way as they span muliple frames
        fixations_by_frame = [[] for x in self.g_pool.timestamps]
        for f in fixations:
            for idx in range(f['start_frame_index'],f['end_frame_index']+1):
                fixations_by_frame[idx].append(f)

        self.g_pool.fixations_by_frame = fixations_by_frame

    @classmethod
    def csv_representation_keys(self):
        return ('id','start_timestamp','duration','start_index','end_frame','norm_pos_x','norm_pos_y','dispersion','avg_pupil_size','confidence')

    @classmethod
    def csv_representation_for_fixation(self, fixation):
        return (
            fixation['id'],
            fixation['timestamp'],
            fixation['duration'],
            fixation['start_frame_index'],
            fixation['end_frame_index'],
            fixation['norm_pos'][0],
            fixation['norm_pos'][1],
            fixation['dispersion'],
            fixation['pupil_diameter'],
            fixation['confidence']
        )

    def export_fixations(self,export_range,export_dir):
        #t
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

        fixations_in_section = chain(*self.g_pool.fixations_by_frame[slice(export_range)])
        fixations_in_section = list(dict([(f['id'],f) for f in fixations_in_section]).values()) #remove duplicates
        fixations_in_section.sort(key=lambda f:f['id'])

        with open(os.path.join(export_dir,'fixations.csv'),'w',encoding='utf-8',newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(self.csv_representation_keys())
            for f in fixations_in_section:
                csv_writer.writerow(self.csv_representation_for_fixation(f))
            logger.info("Created 'fixations.csv' file.")

        with open(os.path.join(export_dir,'fixation_report.csv'),'w',encoding='utf-8',newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(('fixation classifier','Dispersion_Duration'))
            csv_writer.writerow(('max_dispersion','{:0.3f} deg'.format(self.max_dispersion)) )
            csv_writer.writerow(('min_duration','{:0.3f} sec'.format(self.min_duration)) )
            csv_writer.writerow((''))
            csv_writer.writerow(('fixation_count',len(fixations_in_section)))
            logger.info("Created 'fixation_report.csv' file.")


    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        self.last_frame_ts = frame.timestamp
        from player_methods import transparent_circle
        events['fixations'] = self.g_pool.fixations_by_frame[frame.index]
        if self.show_fixations:
            for f in self.g_pool.fixations_by_frame[frame.index]:
                x = int(f['norm_pos'][0]*self.img_size[0])
                y = int((1-f['norm_pos'][1])*self.img_size[1])
                transparent_circle(frame.img, (x,y), radius=f['pix_dispersion']/2, color=(.5, .2, .6, .7), thickness=-1)
                cv2.putText(
                    frame.img,
                    '{:d}'.format(f['id']),
                    (x+20,y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,(255,150,100))
                # cv2.putText(frame.img,'%i - %i'%(f['start_frame_index'],f['end_frame_index']),(x,y), cv2.FONT_HERSHEY_DUPLEX,0.8,(255,150,100))

    def close(self):
        self.alive = False


    def get_init_dict(self):
        return {'max_dispersion': self.max_dispersion, 'min_duration':self.min_duration, 'h_fov':self.h_fov, 'v_fov': self.v_fov,'show_fixations':self.show_fixations}

    def cleanup(self):
        self.deinit_gui()

class Sliding_Window(object):
    """docstring for Sliding_Window"""
    def __init__(self, gaze_data, eye_id, min_duration):
        super().__init__()

        def gp_to_normal_mapping(gazepoint):
            gazepoint_idx,gp = gazepoint
            for dat in gp['base_data']:
                if dat['id'] == eye_id and 'circle_3d' in dat:
                    return (gazepoint_idx, dat['circle_3d']['normal'], dat['timestamp'], dat['diameter'])
            return gazepoint_idx, None, None, None

        mapped   = map(gp_to_normal_mapping, enumerate(gaze_data))
        filtered = filter(lambda x: x[1] != None, mapped)

        # unzip triplets
        self.indices, self.normals, self.timestamps, self.diameters = zip(*filtered)

        if not self.indices:
            raise ValueError('No data found for eye id {:d}'.format(eye_id))
        self.eye_id          = eye_id
        self.min_duration    = min_duration
        self.start_index     = 0
        self.stop_index      = 0
        self.distance_fn     = angle_between_normals
        self.distances       = []
        self._max_dist_idc_c = -1
        self._max_dist_idc_r = -1
        self.make_valid()

    def append_frames(self,n=1):
        if (self.stop_index+n > len(self.normals)):
            raise EOFError('Appending {:d} frames would exceed frame bound.'.format(n))

        new_data = self.pupil_normals(self.stop_index, self.stop_index+n)
        old_incl_new = self.pupil_normals(stop=self.stop_index+n)
        for idx, datum in enumerate(new_data):
            datum_dist, (new_max_val, new_max_idx) = self.calc_dist_one_to_many(datum, old_incl_new[:idx-n])

            if not datum_dist: continue
            if new_max_val >= self.max_distance():
                self._max_dist_idc_c = len(datum_dist)-1
                self._max_dist_idc_r = new_max_idx
            self.distances.append(datum_dist) # add column

        self.stop_index += n

    def pop_frames(self,n=1):
        assert(n>0)

        del self.distances[:n]
        for col in self.distances:
            del col[:n]

        self.start_index += n
        self._max_dist_idc_c -= n
        self._max_dist_idc_r -= n
        self.make_valid()

    def make_valid(self):
        _,stop = self.find_min_frame_range(self.start_index)
        if stop > self.stop_index:
            self.append_frames(n=stop-self.stop_index)

        dm_shape = self.dm_shape
        if len(dm_shape) and len(dm_shape) != dm_shape[-1]:
            raise ValueError('Distance matrix not squared: {0[0]:i}x{0[1]:i}'.format(dm_shape))

    def max_distance(self, ignore_last_frame=False):
        max_dist = 0.
        distance_count = len(self.distances)
        if ignore_last_frame: distance_count -= 1

        if  0 <= self._max_dist_idc_c < distance_count and \
            0 <= self._max_dist_idc_r < distance_count:
            return self.distances[self._max_dist_idc_c][self._max_dist_idc_r]

        for i in range(1, distance_count):
            for j in range(i):
                if  max_dist < self.distances[i][j]:
                    max_dist = self.distances[i][j]
                    if not ignore_last_frame:
                        self._max_dist_idc_c = i
                        self._max_dist_idc_r = j
        return max_dist

    def calc_dist_one_to_many(self, base_vector, data):
        distances = []
        max_dist_val = 0.
        max_dist_idx = -1
        for enum_idx, target_vector in enumerate(data):
            dist = self.distance_fn(base_vector,target_vector)
            distances.append(dist)
            if dist > max_dist_val:
                max_dist_val = dist
                max_dist_idx = enum_idx
        return distances, (max_dist_val, max_dist_idx)

    def find_min_frame_range(self,start_idx=0):
        """Returns minimal frame range for duration `min_duration`
        starting from index `start_idx`

        Args:
            min_duration (float): Description
            start_idx (int, optional): Description

        Returns:
            2 int tuple: range values. See `range()` https://docs.python.org/2/library/functions.html#range
        """
        assert(self.min_duration > 0)
        if start_idx >= len(self.timestamps):
            raise EOFError('Start index for range search out of bounds.')
        ts0 = self.timestamps[start_idx]
        for j in range(start_idx+1, len(self.timestamps)):
            ts1 = self.timestamps[j]
            if abs(ts1 - ts0) > self.min_duration:
                return start_idx, j
        raise EOFError('Could not find a sliding window with minimal lenght of {:.2f}s'.format(self.min_duration))

    def pupil_normals(self,start=None,stop=None):
        start = start or self.start_index
        stop  = stop  or self.stop_index
        return self.normals[start:stop]

    @property
    def slice(self):
        return slice(self.start_index, self.stop_index)

    @property
    def dm_shape(self):
        shape = []
        for col in self.distances:
            shape.append(len(col))
        return tuple(shape)

class Pupil_Angle_3D_Fixation_Detector(Gaze_Position_2D_Fixation_Detector):
    """Fixation detector that uses pupil normal angle for dispersion calculations."""
    def __init__(self, g_pool, max_dispersion = 1.0,min_duration = 0.15,h_fov=78, v_fov=50,show_fixations = False,merge_strategy='shorter_duration'):
        self.dispersion_slider_min = .0
        self.dispersion_slider_max = 5.
        self.dispersion_slider_stp = .1
        self.gaze_data = list(chain(*g_pool.gaze_positions_by_frame))
        self.merge_strategy = merge_strategy
        super().__init__(g_pool, max_dispersion, min_duration, h_fov, v_fov, show_fixations)

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return
        self.last_frame_ts = frame.timestamp
        from player_methods import transparent_circle
        events['fixations'] = self.g_pool.fixations_by_frame[frame.index]
        if self.show_fixations:
            for f in self.g_pool.fixations_by_frame[frame.index]:
                eye_id = f['eye_id']
                x = int(f['norm_pos'][0]*self.img_size[0])
                y = int((1-f['norm_pos'][1])*self.img_size[1])
                transparent_circle(frame.img, (x,y), radius=f['pix_dispersion']/2, color=(.5, .2, .6, .7), thickness=-1)
                cv2.putText(
                    frame.img,
                    '{:d} - eye {:d}'.format(f['id'], eye_id),
                    (x+20,y-5+30*eye_id),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,(255,150,100))
                # cv2.putText(frame.img,'%i - %i'%(f['start_frame_index'],f['end_frame_index']),(x,y), cv2.FONT_HERSHEY_DUPLEX,0.8,(255,150,100))

    @classmethod
    def menu_title(self):
        return 'Pupil Angle Dispersion Fixation Detector'

    @classmethod
    def csv_representation_keys(self):
        return ('id','eye_id','start_timestamp','duration','start_index','end_frame','norm_pos_x','norm_pos_y','dispersion','avg_pupil_size','confidence')

    @classmethod
    def csv_representation_for_fixation(self, fixation):
        return (
            fixation['id'],
            fixation['eye_id'],
            fixation['timestamp'],
            fixation['duration'],
            fixation['start_frame_index'],
            fixation['end_frame_index'],
            fixation['norm_pos'][0],
            fixation['norm_pos'][1],
            fixation['dispersion'],
            fixation['pupil_diameter'],
            fixation['confidence']
        )

    def _classify(self):
        self.fixations = []
        fixs_eye0 = self.fixations_for_eye_id(0)
        fixs_eye1 = self.fixations_for_eye_id(1)
        self.fixations = fixs_eye0 + fixs_eye1

        if not self.fixations:
            logger.error('No fixations could be found. This detector needs 3D pupil data to work. Please try the 2D detector.')
            return

        self.fixations.sort(key=lambda f: f['timestamp'])
        def assign_idx_as_id(id_fix):
            idx, fix = id_fix
            fix['id'] = idx
            return idx
        map(assign_idx_as_id, enumerate(self.fixations))

        fixations_by_frame = [[] for x in self.g_pool.timestamps]
        for f in self.fixations:
            for idx in range(f['start_frame_index'],f['end_frame_index']+1):
                fixations_by_frame[idx].append(f)

        self.g_pool.fixations_by_frame = fixations_by_frame


    def fixations_for_eye_id(self,eye_id):
        t_start = time.time()
        try:
            sw = Sliding_Window(self.gaze_data, eye_id, self.min_duration)
        except ValueError:
            return []

        fix_found = False
        fixations = []

        def add_fixation_for_sliding_window(detect_win):
            dispersion = sw.max_distance(ignore_last_frame=True)
            gaze_selector = itemgetter(*sw.indices[sw.slice])
            fixation_support = gaze_selector(self.gaze_data)
            fix_sup_len = len(fixation_support)
            fixation_centroid = sum([
                    p['norm_pos'][0] for p in fixation_support
                ])/fix_sup_len, sum([
                    p['norm_pos'][1] for p in fixation_support
                ])/fix_sup_len
            confidence = sum(p['confidence'] for p in fixation_support)/fix_sup_len
            avg_pupil_size = sum(sw.diameters[sw.slice])/(sw.stop_index-sw.start_index)
            duration = fixation_support[-1]['timestamp'] - fixation_support[0]['timestamp']

            new_fixation = {
                'id'               : len(fixations),
                'topic'            :'fixation',
                'norm_pos'         :fixation_centroid,
                'base_data'        :fixation_support,
                'duration'         :duration,
                'dispersion'       :np.rad2deg(dispersion),
                'start_frame_index':fixation_support[0]['index'],
                'mid_frame_index'  :fixation_support[int(fix_sup_len/2)]['index'],
                'end_frame_index'  :fixation_support[-1]['index'],
                'pix_dispersion'   :np.rad2deg(dispersion)*self.pix_per_degree,
                'timestamp'        :fixation_support[0]['timestamp'],
                'pupil_diameter'   :avg_pupil_size,
                'confidence'       :confidence,
                'eye_id'           :eye_id
            }
            fixations.append(new_fixation)

        try:
            while True:
                if sw.max_distance() < np.deg2rad(self.max_dispersion):
                    sw.append_frames()
                    fix_found = True
                elif fix_found:
                    add_fixation_for_sliding_window(sw)
                    # move sliding window to the end of fixation
                    sw.pop_frames(n=sw.stop_index-1-sw.start_index)
                    fix_found = False
                else:
                    # move sliding window by one
                    sw.pop_frames(n=1)
        except EOFError:
            if fix_found and sw.stop_index-1 - sw.start_index > 0:
                add_fixation_for_sliding_window(sw)

        total_fixation_time  = sum([f['duration'] for f in fixations])
        total_video_time = self.g_pool.timestamps[-1]-self.g_pool.timestamps[0]
        fixation_count = len(fixations)
        t_stop = time.time()
        logger.info("Detected {} fixations for eye {:d}. Total duration of fixations: {:0.2f}sec total time of video {:0.2f}sec. Took {:.5f}sec to calculate.".format(fixation_count, eye_id, total_fixation_time, total_video_time, t_stop - t_start))
        return fixations


class Detection_Window(object):
    def __init__(self):
        """Data structure to find fixtions

        Holds gaze points and index to corresponding base datum
        Data format: [(gaze_point, base_data_idx),...]
        """
        super().__init__()
        self.distance_fn     = angle_between_normals
        self.gaze_data       = []
        self.pupil_data      = []
        self.distances       = []
        self._max_dist_idc_c = -1
        self._max_dist_idc_r = -1

    def append(self, datum):
        gaze_point, pupil_point = datum
        self.pupil_data.append(pupil_point)
        normal_distances, (new_max_val, new_max_idx) = self.calc_distance(pupil_point, self.pupil_data)

        if new_max_val >= self.max_distance:
            self._max_dist_idc_c = len(normal_distances)-1
            self._max_dist_idc_r = new_max_idx

        self.distances.append(normal_distances)
        self.gaze_data.append(gaze_point)

    def calc_distance(self, base_datum, target_data):
        distances = []
        max_dist_val = 0.
        max_dist_idx = -1
        base_vector = base_datum['circle_3d']['normal']
        for enum_idx, target_datum in enumerate(target_data):
            target_vector = target_datum['circle_3d']['normal']
            dist = self.distance_fn(base_vector,target_vector)
            distances.append(dist)
            if dist > max_dist_val:
                max_dist_val = dist
                max_dist_idx = enum_idx
        return distances, (max_dist_val, max_dist_idx)

    def remove_surplus_data(self, time_constraint):
        if not self.gaze_data: return

        counter = 1
        newest = self.gaze_data[-1]['timestamp']
        for idx, datum in enumerate(self.gaze_data):
            if abs(newest - datum['timestamp']) > time_constraint:
                counter += 1
            else:
                # do not remove datum that breaks the constraint
                counter -= 1
                break
        self.remove_n_datums(counter)

    def remove_n_datums(self, n):
        if n < 0: raise ValueError('`n` needs to be an integer not smaller than zero. `{}` given.'.format(n))
        del self.pupil_data[:n]
        del self.gaze_data[:n]
        del self.distances[:n]
        for col in self.distances:
            del col[:n]

        self._max_dist_idc_c -= n
        self._max_dist_idc_r -= n

    def pop_fixation(self):
        fix_sup_len = len(self.gaze_data)
        fixation_centroid = (
            sum([p['norm_pos'][0] for p in self.gaze_data])/fix_sup_len,
            sum([p['norm_pos'][1] for p in self.gaze_data])/fix_sup_len
        )
        confidence = sum(p['confidence'] for p in self.pupil_data)/fix_sup_len
        avg_diameter = sum(p['diameter'] for p in self.pupil_data)/fix_sup_len
        new_fixation = {
            'topic'            :'fixation',
            'norm_pos'         :fixation_centroid,
            'base_data'        :self.gaze_data,
            'duration'         :self.duration,
            'dispersion'       :self.max_distance_deg,
            'timestamp'        :self.gaze_data[0]['timestamp'],
            'pupil_diameter'   :avg_diameter,
            'confidence'       :confidence,
            'eye_id'           :self.pupil_data[0]['id']
        }

        self.remove_n_datums(len(self.gaze_data))
        return new_fixation

    @property
    def duration(self):
        if not self.gaze_data: return 0.
        t0 = self.gaze_data[ 0]['timestamp']
        t1 = self.gaze_data[-1]['timestamp']
        return t1-t0

    @property
    def max_distance(self):
        distance_count = len(self.distances)
        if (0 <= self._max_dist_idc_c < distance_count and
            0 <= self._max_dist_idc_r < distance_count):
            return self.distances[self._max_dist_idc_c][self._max_dist_idc_r]

        max_dist = 0.
        for i in range(1, distance_count):
            for j in range(i):
                if (max_dist < self.distances[i][j]):
                    max_dist = self.distances[i][j]
                    self._max_dist_idc_c = i
                    self._max_dist_idc_r = j
        return max_dist

    @property
    def max_distance_deg(self):
        return np.rad2deg(self.max_distance)

    def __str__(self):
        return '<{} {:f}sec {:f}deg>'.format(self.__class__.__name__, self.duration, self.max_distance)


class Online_Base_Fixation_Detector(Plugin):
    order = .2
    uniqueness = 'by_base_class'

    def __init__(self, g_pool):
        super().__init__(g_pool)


class Fixation_Detector_3D(Online_Base_Fixation_Detector):
    """docstring for Online_Fixation_Detector_Pupil_Angle_Dispersion_Duration
    """
    def __init__(self, g_pool, max_dispersion=1.0, min_duration=0.15):
        super().__init__(g_pool)
        self.min_duration = min_duration
        self.max_dispersion = max_dispersion
        self.dispersion_slider_min = 0.
        self.dispersion_slider_max = 3.
        self.dispersion_slider_stp = 0.05
        self.detection_windows = [Detection_Window(), Detection_Window()]

    def recent_events(self, events):
        recent = [[],[]]
        events['fixations'] = []
        for gp in events['gaze_positions']:
            for base_idx, base_datum in enumerate(gp['base_data']):
                eye_id = base_datum['id']
                if 'circle_3d' in base_datum:
                    recent[eye_id].append((gp, base_datum))
        for eye_id in range(2):
            detect_win = self.detection_windows[eye_id]
            # add recent gaze positions one by one to make sure we do not overstep
            for datum in recent[eye_id]:
                detect_win.remove_surplus_data(self.min_duration)
                detect_win.append(datum)
                if (detect_win.duration >= self.min_duration and
                    detect_win.max_distance_deg <= self.max_dispersion):
                    new_fixation = detect_win.pop_fixation()
                    events['fixations'].append(new_fixation)

    def init_gui(self):
        def close():
            self.alive = False

        help_str = "Dispersion-based fixation detector. Uses the 3D-model's pupil normal as dispersion measure."
        self.menu = ui.Growing_Menu('3D Fixation Detector')
        self.menu.collapsed = True
        self.menu.append(ui.Button('Close',close))
        self.menu.append(ui.Info_Text(help_str))
        def set_duration(new_value):
            self.min_duration = new_value
            self.notify_all({'subject':'fixations_should_recalculate','delay':1.})
        def set_dispersion(new_value):
            self.max_dispersion = new_value
            self.notify_all({'subject':'fixations_should_recalculate','delay':1.})
        self.menu.append(ui.Slider('min_duration',self,min=0.0,step=0.05,max=1.0,label='Duration threshold',setter=set_duration))
        self.menu.append(ui.Slider('max_dispersion',self,
            min =self.dispersion_slider_min,
            step=self.dispersion_slider_stp,
            max =self.dispersion_slider_max,
            label='Dispersion threshold',
            setter=set_dispersion))
        self.g_pool.sidebar.append(self.menu)

    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None

    def get_init_dict(self):
        return {'max_dispersion': self.max_dispersion, 'min_duration':self.min_duration}

    def cleanup(self):
        self.deinit_gui()
