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

class Dummy_Gaze_Mapper(Gaze_Mapping_Plugin):
    """docstring for Dummy_Gaze_Mapper"""
    def __init__(self, g_pool):
        super(Dummy_Gaze_Mapper, self).__init__(g_pool)

    def update(self,frame,events):
        gaze_pts = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                gaze_pts.append({'norm_pos':p['norm_pos'][:],'confidence':p['confidence'],'timestamp':p['timestamp']})

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
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp']})

        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {'params':self.params}


class Volumetric_Gaze_Mapper(Gaze_Mapping_Plugin):
    def __init__(self,g_pool,params):
        super(Volumetric_Gaze_Mapper, self).__init__(g_pool)
        self.params = params

    def update(self,frame,events):
        gaze_pts = []
        raise NotImplementedError
        events['gaze_positions'] = gaze_pts

    def get_init_dict(self):
        return {'params':self.params}

class Binocular_Gaze_Mapper(Gaze_Mapping_Plugin):
    def __init__(self, g_pool,params):
        super(Binocular_Gaze_Mapper, self).__init__(g_pool)
        self.params = params
        self.map_fns = (make_map_function(*self.params[0:3]),make_map_function(*self.params[3:6]))

    def update(self,frame,events):
        gaze_pts = []
        gaze_mono_pts = [[],[]]
        
        for p in events['pupil_positions']:
            if p['confidence'] > self.g_pool.pupil_confidence_threshold:
                eye_id = p['id']
                gaze_point = self.map_fns[eye_id](p['norm_pos'])
                gaze_mono_pts[eye_id].append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp']})
        
        # Pair gaze positions and compute means
        i = 0
        j = 0
        while i < len(gaze_mono_pts[0]) and j < len(gaze_mono_pts[1]):
            gaze_0 = gaze_mono_pts[0][i]
            gaze_1 = gaze_mono_pts[1][j]
            diff = gaze_0['timestamp'] - gaze_1['timestamp']
            if abs(diff) <= 1/15.: #assuming 30fps + slack
                x_0, y_0 = gaze_0['norm_pos']
                x_1, y_1 = gaze_1['norm_pos']
                gaze_point = ((x_0+x_1)/2,(y_0+y_1)/2)
                confidence = min(gaze_0['confidence'], gaze_1['confidence'])
                timestamp = max(gaze_0['timestamp'], gaze_1['timestamp'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':confidence,'timestamp':timestamp})
                i += 1
                j += 1
            elif diff > 0:
                j += 1
            else:
                i += 1
        
        events['gaze_positions'] = gaze_pts
        events['gaze_eye_0'] = gaze_mono_pts[0]
        events['gaze_eye_1'] = gaze_mono_pts[1]

    def get_init_dict(self):
        return {'params':self.params}
