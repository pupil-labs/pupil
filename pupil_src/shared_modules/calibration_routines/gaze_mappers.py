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
from pyglui import ui


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
    def __init__(self, g_pool,params,params_eye0,params_eye1):
        super(Gaze_Mapping_Plugin, self).__init__(g_pool)
        self.params = params
        self.params_eye0 = params_eye0
        self.params_eye1 = params_eye1
        self.multivariate = True
        self.map_fn = make_map_function(*self.params)
        self.map_fn_fallback = []
        self.map_fn_fallback.append(make_map_function(*self.params_eye0))
        self.map_fn_fallback.append(make_map_function(*self.params_eye1))

    def init_gui(self):
        self.menu = ui.Growing_Menu('Binocular Gaze Mapping')
        self.g_pool.sidebar.insert(3,self.menu)
        self.menu.append(ui.Switch('multivariate',self,on_val=True,off_val=False,label='Multivariate Mode'))

    def update(self,frame,events):
        
        pupil_pts_0 = []
        pupil_pts_1 = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                if p['id'] == 0:
                    pupil_pts_0.append(p)
                else:
                    pupil_pts_1.append(p)

        # try binocular mapping (needs at least 1 pupil position in each list)
        gaze_pts = []
        if len(pupil_pts_0) > 0 and len(pupil_pts_1) > 0:
            gaze_pts = self._map_binocular(pupil_pts_0, pupil_pts_1, self.multivariate)
        # fallback to monocular if something went wrong
        else:
            for p in pupil_pts_0:
                gaze_point = self.map_fn_fallback[0](p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})
            for p in pupil_pts_1:
                gaze_point = self.map_fn_fallback[1](p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'],'base':[p]})
            
        events['gaze_positions'] = gaze_pts
        
    def _map_binocular(self, pupil_pts_0, pupil_pts_1,multivariate=True):
        # maps gaze with binocular mapping
        # requires each list to contain at least one item!
        # returns 1 gaze point at minimum
        gaze_pts = []
        p0 = pupil_pts_0.pop(0)
        p1 = pupil_pts_1.pop(0)
        while True:
            if multivariate:
                gaze_point = self.map_fn(p0['norm_pos'], p1['norm_pos'])
            else:
                gaze_point_eye0 = self.map_fn_fallback[0](p0['norm_pos'])
                gaze_point_eye1 = self.map_fn_fallback[1](p1['norm_pos'])
                gaze_point = (gaze_point_eye0[0] + gaze_point_eye1[0])/2. , (gaze_point_eye0[1] + gaze_point_eye1[1])/2. 
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
        
        return gaze_pts
    
    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None
            
    def cleanup(self):
        self.deinit_gui()

    def get_init_dict(self):
        return {'params':self.params, 'params_eye0':self.params_eye0, 'params_eye1':self.params_eye1}