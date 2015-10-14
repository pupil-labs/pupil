'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2015  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

from plugin import Gaze_Mapping_Plugin
from calibrate import make_map_function
from copy import deepcopy
import numpy as np


class Dummy_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Dummy_Gaze_Mapper"""
    def __init__(self, g_pool):
        super(Dummy_Gaze_Mapper, self).__init__(g_pool)

    def update(self,frame,events):
        gaze_pts = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                gaze_pts.append({'norm_pos':p['norm_pos'][:],'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})

        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {}


class Simple_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Simple_Gaze_Mapper"""
    def __init__(self, g_pool,params):
        super(Simple_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.map_fn = make_map_function(*self.params)

    def update(self,frame,events):
        gaze_pts = []

        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                gaze_point = self.map_fn(p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})

        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {'params':self.params}


    # def map_gaze_offline(self,pupil_positions):
    #     min_confidence = self.g_pool.pupil_confidence_threshold
    #     gaze_pts = deepcopy(pupil_positions)
    #     norm_pos = np.array([p['norm_pos'] for p in gaze_pts])
    #     norm_pos = self.map_fn(norm_pos.T)
    #     for n in range(len(gaze_pts)):
    #         gaze_pts[n]['norm_pos'] = norm_pos[0][n],norm_pos[1][n]
    #         gaze_pts[n]['base'] = [pupil_positions[n]]
    #     gaze_pts = filter(lambda g: g['confidence']> min_confidence,gaze_pts)
    #     return gaze_pts


class Volumetric_Gaze_Mapper(Gaze_Mapping_Plugin):
    def __init__(self,g_pool,params):
        super(Volumetric_Gaze_Mapper, self).__init__(g_pool)
        self.params = params

    def update(self,frame,events):
        gaze_pts = []
        raise NotImplementedError()
        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {'params':self.params}

class Bilateral_Gaze_Mapper(Gaze_Mapping_Plugin):
    def __init__(self, g_pool,params):
        super(Gaze_Mapping_Plugin, self).__init__(g_pool)
        self.params = params
        self.map_fn = make_map_function(*self.params)

    def update(self,frame,events):
        
        pupil_pts_0 = []
        pupil_pts_1 = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                if p['id'] == 0:
                    pupil_pts_0.append(p)
                else:
                    pupil_pts_1.append(p)
                
        if len(pupil_pts_0) <= 0 or len(pupil_pts_1) <= 0:
            # TODO: fallback to monocular if possible
            events['gaze_positions'] = []
            return


        gaze_pts = []
        p0 = pupil_pts_0.pop(0)
        p1 = pupil_pts_1.pop(0)
        while True:
            gaze_point = self.map_fn(p0['norm_pos'], p1['norm_pos'])
            confidence = (p0['confidence'] + p1['confidence'])/2.
            ts = (p0['timestamp'] + p1['timestamp'])/2.
            gaze_pts.append({'norm_pos':gaze_point,'confidence':confidence,'timestamp':ts,'base':[p0, p1]})
            
            # keep sample with higher timestamp and increase the one with lower timestamp
            if p0['timestamp'] <= p1['timestamp'] and pupil_pts_0:
                p0 = pupil_pts_0.pop(0)
                continue
            elif p1['timestamp'] <= p0['timestamp'] and pupil_pts_1:
                p1 = pupil_pts_1.pop(0)
                continue
            elif pupil_pts_0 and not pupil_pts_1:
                p0 = pupil_pts_0.pop(0)
            elif pupil_pts_1 and not pupil_pts_0:
                p1 = pupil_pts_1.pop(0)
            else:
                break            

        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {'params':self.params}